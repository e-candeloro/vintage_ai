from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    api_base_path: str = "/"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
