# One image, two entrypoints. The API and the worker run the same code at
# different entry points, so they cannot drift in library versions, and the
# difference between them is the command and the network they sit on.
#
# Non root, per SECURITY.md control 9. The application never writes to its own
# filesystem, so the container runs read only in compose with a tmpfs for /tmp.

FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/srv

WORKDIR /srv

# Dependencies first, so a source change does not reinstall the world.
COPY pyproject.toml ./
RUN python -m pip install --upgrade pip \
 && python -m pip install "." \
 && python -m pip install gunicorn

COPY app ./app
COPY pipeline ./pipeline

# A fixed uid so a bind mounted volume has predictable ownership, and no shell
# for the account that runs the process.
RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin provenance
USER 10001

# gthread workers: ARCHITECTURE.md, because the request path is IO bound.
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--worker-class", "gthread", \
     "--workers", "2", "--threads", "4", "--timeout", "30", \
     "--access-logfile", "-", "app:create_app()"]
