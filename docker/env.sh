# Source from an interactive `docker exec` so `python`/`pip`/`torchrun` are the
# serving venv, not Ubuntu's package-less python3:
#   source /app/docker/env.sh
# Compose also prepends this via PATH on recreate. Training deps still live in
# /cache/venv-dflash2 — never `pip install nemo-automodel` into /app/venv.
if [ -x /app/venv/bin/python ]; then
  export PATH="/app/docker/bin:/app/venv/bin:$PATH"
fi
export VIRTUAL_ENV="${VIRTUAL_ENV:-/app/venv}"
