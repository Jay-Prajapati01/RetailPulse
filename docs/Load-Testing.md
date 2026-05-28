# Load testing RetailPulse

This runbook covers the Day 27 validation harness.

## Install dev dependencies

```powershell
python -m pip install -r requirements-dev.txt
```

## Run Locust

Start the dashboard first, then run:

```powershell
locust -f performance/locustfile.py --host http://localhost:8501
```

Open the Locust UI in your browser and start a small swarm to validate:

- dashboard responsiveness
- page navigation
- session stability

## Notes

- Keep the first run small, then scale users gradually.
- Use this only against a local or disposable environment.
