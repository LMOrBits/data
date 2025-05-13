from enum import Enum
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from lakefs import Repository, Branch, Commit
from lakefs.client import Client
import lakefs
from lakefs_spec import LakeFSFileSystem
import os
from pydantic import BaseModel
from loguru import logger


class DatasetType(Enum):
    RAW = "raw"
    BRONZE = "bronze"
    SILVER = "silver"
    GOLD = "gold"

class MLDatasetType(Enum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"
    

@dataclass
class LakeFSCredentials:
    """Credentials for LakeFS authentication"""
    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    namespace: str

    def __post_init__(self):
        os.environ["LAKECTL_SERVER_ENDPOINT_URL"] = self.endpoint_url
        os.environ["LAKECTL_CREDENTIALS_ACCESS_KEY_ID"] = self.access_key_id
        os.environ["LAKECTL_CREDENTIALS_SECRET_ACCESS_KEY"] = self.secret_access_key
        os.environ["LAKECTL_NAMESPACE"] = self.namespace
        if not self.namespace.startswith(('gs://', 's3://', 'azure://')):
            raise ValueError("namespace must start with 'gs://', 's3://' or 'azure://'")
    
    @classmethod
    def from_env(cls) -> 'LakeFSCredentials':
        """Create credentials from environment variables"""
        required_vars = {
            "LAKECTL_SERVER_ENDPOINT_URL": "endpoint_url",
            "LAKECTL_CREDENTIALS_ACCESS_KEY_ID": "access_key_id",
            "LAKECTL_CREDENTIALS_SECRET_ACCESS_KEY": "secret_access_key",
            "LAKECTL_NAMESPACE": "namespace"
        }
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        return cls(**{v: os.getenv(k) for k, v in required_vars.items()})

@dataclass
class StorageConfig:
    """Configuration for storage operations"""
    namespace: str
    project_name: str = "lakefs"

    def __post_init__(self):
        """Validate configuration after initialization"""
        if not self.namespace.startswith(('gs://', 's3://', 'azure://')):
            raise ValueError("namespace must start with 'gs://', 's3://' or 'azure://'")
    
    def build_path(self, repo_name: str, branch_name: Optional[str] = None) -> str:
        """Build storage path for repository"""
        base_path = f"{self.namespace}/{self.project_name}/{repo_name}"
        return f"{base_path}/{branch_name}" if branch_name else base_path

class BranchManager:
    """Manages branch operations"""
    def __init__(self, default_branch: str = "main"):
        self._current_branch = default_branch

    @property
    def current_branch(self) -> str:
        return self._current_branch

    @current_branch.setter
    def current_branch(self, branch_name: str):
        if not branch_name:
            raise ValueError("Branch name cannot be empty")
        self._current_branch = branch_name

    def get_or_create(self, repo: Repository, branch_name: str, source_branch: str) -> Branch:
        """Create or get an existing branch in a repository"""
        self.current_branch = branch_name
        try:
            if branch_name in list(repo.branches()):
                print(f"Branch {branch_name} already exists")
                return repo.branch(branch_name)
            else:
                print(f"Branch {branch_name} does not exist, creating it now")
                branch = repo.branch(branch_name).create(source_branch)
            return branch
        except Exception as e:
            print(f"while creating branch {branch_name} from {source_branch} we got an error {e}") 
            

    def list_all(self, repo: Repository) -> List[str]:
        """List all branches in a repository"""
        return [branch.id for branch in repo.branches.list()]

    def get_status(self, repo: Repository, branch_name: str, default_branch: str) -> Dict:
        """Get the status of a branch including commits behind/ahead of main"""
        branch = repo.branch(branch_name)
        diff = branch.diff(default_branch)
        return {
            "branch": branch_name,
            "commits_behind": len(diff.commits_behind),
            "commits_ahead": len(diff.commits_ahead),
            "changes": len(diff.changes)
        }

class RepositoryManager:
    """Manages repository operations"""
    def __init__(self, repo_name: str , storage_config: StorageConfig, branch_manager: BranchManager):
        self.storage_config = storage_config
        self.branch_manager = branch_manager
        self.repo_name = repo_name
        self.repo = None
    
    def storage_path(self,branch_name: str = "main"):
        return self.storage_config.build_path(self.repo_name,branch_name)
    
    def get_or_create(self) -> Repository:
        """Create or get an existing repository"""
        try:
            repo = lakefs.repository(self.repo_name)
            print(f"Found existing repo {repo.id} using storage namespace {repo.properties.storage_namespace}")
            self.repo = repo
            return repo
        except Exception as e:
            print(f"Repository {self.repo_name} does not exist, creating it now.")
            try:
                storage_path = self.storage_config.build_path(self.repo_name)
                repo = lakefs.Repository(self.repo_name).create(
                    storage_namespace=storage_path,
                    default_branch=self.branch_manager.current_branch,
                    exist_ok=True
                )
                print(f"Created new repo {repo.id} using storage namespace {repo.properties.storage_namespace}")
                self.repo = repo
                return repo
            except Exception as e:
                raise Exception(f"Error creating repo {self.repo_name}. Error: {str(e)}")

class LakeFSClient:
    """Main client for LakeFS operations"""
    def __init__(self, repo_name: str, storage_config: StorageConfig, credentials: LakeFSCredentials, 
                  default_branch: str = "main"):
        print(credentials)
        self.branch_manager = BranchManager(default_branch)
        self.repo_manager = RepositoryManager(repo_name, storage_config, self.branch_manager)
        self.fs: LakeFSFileSystem = LakeFSFileSystem(
            host=credentials.endpoint_url,
            username=credentials.access_key_id,
            password=credentials.secret_access_key
        )
    @property
    def path_simple(self):
        return f"lakefs://{self.repo_manager.repo_name}"
    @property
    def path(self):
        return f"lakefs://{self.repo_manager.repo_name}/{self.branch_manager.current_branch}"
    def get_or_create_branch(self,  branch_name: str = "main",
                                     source_branch: Optional[str] = None) -> Tuple[Repository, Branch]:
        """Create or get both repository and branch"""
        self.repo = self.repo_manager.get_or_create()
        self.branch = self.branch_manager.get_or_create(
            self.repo, 
            branch_name, 
            source_branch or self.branch_manager.current_branch
        )
        return self.repo, self.branch

    @classmethod
    def client_factory(cls, credentials: LakeFSCredentials,
                       repo_name: str,
                       branch_name: str = "main",
                       source_branch: str = "main") -> 'LakeFSClient':
        storage_config = StorageConfig(namespace=credentials.namespace)
        lakefs_client = cls(repo_name, storage_config, credentials)
        lakefs_client.get_or_create_branch(branch_name=branch_name , source_branch=source_branch)
        return lakefs_client


class LakeFsDatasetModel(BaseModel):
    dataset_type: DatasetType
    directory: str
    def get_path(self):
        return f"{self.dataset_type.value}/{self.directory}"

class LakeFsDataset:
    def __init__(self,
                 credentials: LakeFSCredentials,
                 dataset_type: DatasetType, 
                 directory: str, 
                 project_name: str, 
                 branch_name: str = "main",
                 source_branch: str = "main"):
        self.credentials = credentials
        self.dataset = LakeFsDatasetModel(dataset_type=dataset_type, directory=directory)
        self.lakefs_client = LakeFSClient.client_factory(credentials, repo_name= project_name, branch_name= branch_name , source_branch=source_branch)
    
    @property
    def branch(self):
        return self.lakefs_client.branch_manager.current_branch
    @branch.setter
    def branch(self, branch_name: str):
        self.lakefs_client.branch_manager.current_branch = branch_name
    @property
    def repo(self):
        return self.lakefs_client.repo_manager.repo_name
    @repo.setter
    def repo(self, repo_name: str):
        self.lakefs_client.repo_manager.repo_name = repo_name
        
    

    def get_path(self):
        return f"{self.lakefs_client.path}/{self.dataset.get_path()}"
    
    @property
    def full_path(self):
        return f"{self.lakefs_client.path}/{self.dataset.get_path()}"
    
    def ls_folders(self,path:str=""):
        list_of_folders = self.lakefs_client.fs.glob(self.full_path + path + "/*/" , maxdepth=1)
        if list_of_folders:
            list_of_folders = [folder.split("/")[-2] for folder in list_of_folders]
        return list_of_folders
    
    def ls(self,path:str):
        logger.info(f"Listing files in {self.full_path + path}")
        list_of_files = self.lakefs_client.fs.glob(self.full_path + path , maxdepth=3 , refresh=True)
        if list_of_files:
            list_of_files = ["lakefs://" + file for file in list_of_files]
        return list_of_files
   
    def load_data_files(self,  main_dir: str = "" , format_path :str = "/**/*.parquet") -> dict[str,list[str]]:
        folders : list[str] = self.ls_folders(path=main_dir)
        data_files : dict[str,str] = {folder:self.ls(path=f"/{folder}{format_path}") for folder in folders}
        return data_files 
    

class LakeFsEmbeding:
    def __init__(self,
                 credentials: LakeFSCredentials,
                 project_name: str, 
                 branch_name: str = "main",
                 source_branch: str = "main",
                 prefix: str = "vectordb"):
        self.credentials = credentials
        self.lakefs_client = LakeFSClient.client_factory(credentials, repo_name= project_name, branch_name= branch_name , source_branch=source_branch)
        self._prefix = prefix
    
    @property
    def branch(self):
        return self.lakefs_client.branch_manager.current_branch
    @branch.setter
    def branch(self, branch_name: str):
        self.lakefs_client.branch_manager.current_branch = branch_name
    @property
    def repo(self):
        return self.lakefs_client.repo_manager.repo_name
    @repo.setter
    def repo(self, repo_name: str):
        self.lakefs_client.repo_manager.repo_name = repo_name
    @property
    def prefix(self):
        return self._prefix

    def get_path(self,id:Optional[str] = None):
        if id:
            return f"{self.lakefs_client.path_simple}/{id}/{self.prefix}"
        return f"{self.lakefs_client.path}/{self.prefix}"
    
    @property
    def full_path(self):
        return f"{self.lakefs_client.path}"
    
    