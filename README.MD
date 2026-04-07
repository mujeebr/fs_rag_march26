# FS RAG (Streamlit)

## Local development (no Kubernetes)

1. Create and activate a virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`

---

## Run on Minikube with an image from Docker Desktop

Kubernetes inside Minikube does **not** see images in Docker Desktop by default. It uses the cluster’s own image store. If you only built `my-app:latest` in Docker Desktop, the pod fails with **ErrImagePull** / **ImagePullBackOff** until that image is available inside Minikube.

### 1. Build the image in Docker Desktop (if you have not already)

From the project root (where the `Dockerfile` is):

```bash
docker build -t my-app:latest .
```

Confirm the image exists locally:

```bash
docker images my-app
```

### 2. Load the image from Docker Desktop into Minikube

This copies the image from your Mac’s Docker into the Minikube node.

**Before any load:** If you ever ran `eval "$(minikube docker-env)"`, your `docker` CLI points at **Minikube’s** Docker, not Docker Desktop. Unset it first (run this whenever you are unsure):

```bash
eval "$(minikube docker-env -u)"
```

**Run one command at a time** (press Enter after each). Do not paste `minikube image load` and `rm` on the same line—that can break the path (for example `/tmp/my-app.tarrm`) or delete the tar before load finishes.

#### Method A — load straight from the Docker daemon

```bash
minikube image load my-app:latest --daemon=true
```

The first load can take **several minutes** with **no progress output** (ML-related images are often large). That is normal; you can watch Activity Monitor for `docker` / `minikube` activity.

#### Method B — tar file (recommended if Method A hangs or you want a clear checkpoint)

1. Ensure Docker Desktop is selected (see “Before any load” above).

2. Export the image to a tar (wait until the shell prompt returns; large images take time):

   ```bash
   docker save my-app:latest -o /tmp/my-app.tar
   ```

3. Load that archive into Minikube (again, may take several minutes with no spinner):

   ```bash
   minikube image load /tmp/my-app.tar
   ```

4. After the load succeeds, remove the tar to free disk space:

   ```bash
   rm /tmp/my-app.tar
   ```

#### Method C — build inside Minikube (no Docker Desktop image needed)

```bash
eval "$(minikube docker-env)"
docker build -t my-app:latest .
eval "$(minikube docker-env -u)"
```

### 3. Apply the pod and service

```bash
kubectl apply -f pod.yaml
```

### 4. Wait until the pod is ready

```bash
kubectl get pods -w
```

You should see `my-app` with `READY` `1/1` and `STATUS` `Running`.

If the pod was already stuck from a failed pull, delete it so Kubernetes creates a new one after the image is loaded:

```bash
kubectl delete pod my-app
kubectl apply -f pod.yaml
```

### 5. Open the app in the browser

```bash
minikube service my-app --url
```

Open the printed URL (Streamlit listens on port **8501**; the Service maps it to NodePort **30080**).

**Without a Service** (pod only), use port-forward:

```bash
kubectl port-forward pod/my-app 8501:8501
```

Then open `http://localhost:8501`.

### 6. Troubleshooting ErrImagePull

| Check | Command / action |
|--------|------------------|
| Image on host | `docker images my-app` |
| Image inside Minikube | `minikube ssh -- docker images my-app` (or `minikube image ls \| grep my-app` on newer Minikube) |
| Pod events | `kubectl describe pod my-app` |
| After fixing image | `kubectl delete pod my-app` then `kubectl apply -f pod.yaml` |

### 7. `minikube image load` looks stuck

1. **Unset Minikube’s Docker env** (common cause): `eval "$(minikube docker-env -u)"`, then try **Method A** again (`--daemon=true`).
2. **Long transfer:** big images produce no spinner; wait several minutes or switch to **Method B** (tar) in section 2.
3. **Wrong path / command glued together:** use exactly `/tmp/my-app.tar` and run `docker save`, then `minikube image load /tmp/my-app.tar`, then `rm` as **separate** lines.
4. **Confirm Docker Desktop has the image** (with docker-env unset): `docker images my-app:latest`.

Ensure `pod.yaml` uses `imagePullPolicy: IfNotPresent` for local images (already set there). For images in a registry only, use `imagePullPolicy: Always` and a full registry name.

### Cleanup

```bash
kubectl delete -f pod.yaml
```
