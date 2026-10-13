"""Unit coverage for API-002 password and JWT primitives."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

import jwt
import pytest

from app.core.security import (
    JWT_ALGORITHM,
    InvalidAccessTokenError,
    PasswordManager,
    TokenManager,
)

SECRET = "deterministic-test-secret-with-sufficient-entropy"


def test_password_hash_is_not_plaintext_and_verifies_only_correct_password() -> None:
    passwords = PasswordManager()

    password_hash = passwords.hash("correct horse battery staple")

    assert password_hash != "correct horse battery staple"
    assert password_hash.startswith("$argon2")
    assert passwords.verify("correct horse battery staple", password_hash) is True
    assert passwords.verify("wrong password", password_hash) is False


def test_jwt_round_trip_has_fixed_algorithm_uuid_subject_and_expiration() -> None:
    now = datetime.now(UTC).replace(microsecond=0)
    user_id = uuid4()
    tokens = TokenManager(secret_key=SECRET, lifetime_minutes=30, clock=lambda: now)

    token = tokens.create(user_id)
    claims = jwt.decode(token, SECRET, algorithms=[JWT_ALGORITHM])

    assert tokens.decode_subject(token) == user_id
    assert jwt.get_unverified_header(token)["alg"] == JWT_ALGORITHM
    assert claims["sub"] == str(user_id)
    assert claims["exp"] - claims["iat"] == 1_800
    assert tokens.expires_in_seconds == 1_800
    assert set(claims) == {"sub", "iat", "exp"}


def test_decoder_rejects_non_frozen_algorithm() -> None:
    user_id = uuid4()
    now = datetime.now(UTC)
    token = jwt.encode(
        {"sub": str(user_id), "iat": now, "exp": now + timedelta(minutes=5)},
        SECRET,
        algorithm="HS384",
    )

    with pytest.raises(InvalidAccessTokenError):
        TokenManager(secret_key=SECRET).decode_subject(token)


def test_decoder_rejects_expired_token() -> None:
    now = datetime.now(UTC)
    token = jwt.encode(
        {
            "sub": str(uuid4()),
            "iat": now - timedelta(minutes=2),
            "exp": now - timedelta(minutes=1),
        },
        SECRET,
        algorithm=JWT_ALGORITHM,
    )

    with pytest.raises(InvalidAccessTokenError):
        TokenManager(secret_key=SECRET).decode_subject(token)


def test_decoder_rejects_invalid_signature_and_invalid_subject() -> None:
    now = datetime.now(UTC)
    valid_claim_times = {"iat": now, "exp": now + timedelta(minutes=5)}
    wrong_signature = jwt.encode(
        {"sub": str(uuid4()), **valid_claim_times},
        "different-test-secret-with-sufficient-entropy",
        algorithm=JWT_ALGORITHM,
    )
    invalid_subject = jwt.encode(
        {"sub": "not-a-uuid", **valid_claim_times},
        SECRET,
        algorithm=JWT_ALGORITHM,
    )
    tokens = TokenManager(secret_key=SECRET)

    with pytest.raises(InvalidAccessTokenError):
        tokens.decode_subject(wrong_signature)
    with pytest.raises(InvalidAccessTokenError):
        tokens.decode_subject(invalid_subject)
