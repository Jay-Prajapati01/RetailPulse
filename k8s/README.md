# RetailPulse Kubernetes

This folder contains a local Kubernetes scaffold for Minikube or kind.

## Apply order

```powershell
kubectl apply -k k8s
```

Or apply manually with init sequencing:

```powershell
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/secret.template.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/persistentvolumeclaim.yaml
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/mlflow-deployment.yaml
kubectl apply -f k8s/streamlit-deployment.yaml
kubectl apply -f k8s/airflow-init-job.yaml
kubectl wait --for=condition=complete --timeout=300s job/airflow-init -n retailpulse
kubectl apply -f k8s/airflow-deployment.yaml
kubectl apply -f k8s/hpa.yaml
kubectl apply -f k8s/pdb.yaml
kubectl apply -f k8s/networkpolicy.yaml
kubectl apply -f k8s/ingress.yaml
```

## Notes

- Replace `retailpulse-*:latest` images with locally built images or push to a registry such as GHCR.
- `secret.template.yaml` is a template. Copy it to a real secret manifest before applying.
- For Minikube ingress, enable the ingress addon and map `retailpulse.local` to your cluster IP.
- For HPA support in Minikube, enable metrics-server: `minikube addons enable metrics-server`.
