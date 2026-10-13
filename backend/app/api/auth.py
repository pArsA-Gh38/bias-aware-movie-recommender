"""HTTP routes for the bounded local authentication slice."""

from collections.abc import Callable
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.auth import (
    AuthenticationFailedError,
    AuthRepositoryUnavailableError,
    AuthService,
    AuthUser,
    DuplicateEmailError,
)
from app.schemas.auth import (
    MAX_PASSWORD_LENGTH,
    MIN_PASSWORD_LENGTH,
    CredentialsRequest,
    TokenResponse,
    UserResponse,
)

AUTHENTICATION_FAILED = "Could not validate credentials"
AUTHENTICATION_UNAVAILABLE = "Authentication service unavailable"
DUPLICATE_EMAIL = "Email is already registered"

CurrentUserDependency = Callable[..., AuthUser]


def unauthorized() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=AUTHENTICATION_FAILED,
        headers={"WWW-Authenticate": "Bearer"},
    )


def create_current_user_dependency(
    auth_service: AuthService | None,
) -> CurrentUserDependency:
    """Build the shared API-002 bearer-to-current-user boundary."""
    bearer = HTTPBearer(auto_error=False)

    def current_user(
        credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> AuthUser:
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise unauthorized()
        if auth_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=AUTHENTICATION_UNAVAILABLE,
            )
        try:
            return auth_service.current_user(credentials.credentials)
        except AuthenticationFailedError as exc:
            raise unauthorized() from exc
        except AuthRepositoryUnavailableError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=AUTHENTICATION_UNAVAILABLE,
            ) from exc

    return current_user


def create_auth_router(
    auth_service: AuthService | None,
    current_user_dependency: CurrentUserDependency | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/auth", tags=["auth"])
    resolve_current_user = current_user_dependency or create_current_user_dependency(
        auth_service
    )

    def require_service() -> AuthService:
        if auth_service is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=AUTHENTICATION_UNAVAILABLE,
            )
        return auth_service

    def unavailable() -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=AUTHENTICATION_UNAVAILABLE,
        )

    def validate_password_policy(password: str) -> None:
        if not MIN_PASSWORD_LENGTH <= len(password) <= MAX_PASSWORD_LENGTH:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                detail="Password must be between 8 and 128 characters",
            )

    @router.post(
        "/register",
        response_model=UserResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def register(credentials: CredentialsRequest) -> UserResponse:
        validate_password_policy(credentials.password)
        try:
            user = require_service().register(
                email=str(credentials.email), password=credentials.password
            )
        except DuplicateEmailError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=DUPLICATE_EMAIL
            ) from exc
        except AuthRepositoryUnavailableError as exc:
            raise unavailable() from exc
        return UserResponse.model_validate(user, from_attributes=True)

    @router.post("/login", response_model=TokenResponse)
    def login(credentials: CredentialsRequest) -> TokenResponse:
        validate_password_policy(credentials.password)
        try:
            token = require_service().login(
                email=str(credentials.email), password=credentials.password
            )
        except AuthenticationFailedError as exc:
            raise unauthorized() from exc
        except AuthRepositoryUnavailableError as exc:
            raise unavailable() from exc
        return TokenResponse(access_token=token.value, expires_in=token.expires_in)

    @router.get("/me", response_model=UserResponse)
    def me(
        user: Annotated[AuthUser, Depends(resolve_current_user)],
    ) -> UserResponse:
        return UserResponse.model_validate(user, from_attributes=True)

    return router
