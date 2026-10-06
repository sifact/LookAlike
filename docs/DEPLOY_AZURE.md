# Deploy to Azure with a container and Kubernetes (Portal, no Azure CLI)

You build the container image on your Mac with Docker. Everything on the Azure side is
done by clicking through the [Azure Portal](https://portal.azure.com).

```
your Mac                         Azure
┌───────────────┐  docker push  ┌──────────────────────────┐
│ docker build  │ ────────────> │ Container Registry (ACR) │
└───────────────┘               └────────────┬─────────────┘
                                             │ image pull
                                ┌────────────▼─────────────┐   public IP   ┌─────────┐
                                │ Kubernetes cluster (AKS) │ <──────────── │ browser │
                                │  Deployment: 1 pod       │   Service     └─────────┘
                                │  (FastAPI + React UI)    │  LoadBalancer
                                └──────────────────────────┘
```

| Piece | What it is |
|---|---|
| **Container image** | One package with Python, PyTorch, the code, the web UI, the trained model, the index and the image thumbnails ([Dockerfile](../Dockerfile)) |
| **Azure Container Registry (ACR)** | Private storage for container images, like a private Docker Hub |
| **Azure Kubernetes Service (AKS)** | Runs and restarts containers on virtual machines ("nodes") |
| **Deployment** | Tells Kubernetes to keep 1 copy (pod) of the image running ([visual-search.yaml](../deploy/k8s/visual-search.yaml)) |
| **Service** | A stable internal address for the pod |
| **Ingress (NGINX) + cert-manager** | The public HTTPS front door, with a free Let's Encrypt certificate on a free `*.cloudapp.azure.com` name (step 6) |

**Your subscription's limits:** the free trial allows 4 vCPUs per region. Your existing
`spam-email-validation-aks` cluster already uses 2, so the new cluster must be **1 node with
2 vCPUs**. The app needs about 1–1.5 GB of RAM, so that's enough.

---

## Step 1: Prepare the bundle (Mac terminal)

The image needs the trained model, the index, the catalog and the images. This collects
them into `deploy/bundle/` and shrinks the images to 320 px thumbnails, about 830 MB in total.

```bash
cd ~/Documents/Dev/ai-porjects/lookalike
uv run python deploy/make_bundle.py --index artifacts/indexes/interactions_1790965435_hnsw
```

## Step 2: Create the container registry (Portal)

1. Search **"Container registries"** → **+ Create**.
2. **Basics** tab:
   - Subscription: *Azure subscription 1*
   - Resource group: **visual-search-rg** (already exists)
   - Registry name: a globally unique name, letters and numbers only, e.g. **visualsearchsifat**
   - Location: **East US**
   - Pricing plan: **Basic**
3. **Review + create** → **Create**, then wait about 1 minute → **Go to resource**.
4. Left menu **Settings → Access keys** → turn on **Admin user**.
   Note the **Login server** (`visualsearchsifat.azurecr.io`), the **Username** and **password**.
   You'll need them in step 3.

## Step 3: Build and push the image (Mac terminal, Docker)

AKS nodes are Intel/AMD (`amd64`), but your Mac is ARM, so Docker must build an
`amd64` image. Colima can do that quickly with Rosetta:

```bash
colima stop 2>/dev/null; colima start --vm-type vz --vz-rosetta --memory 6

REG=visualsearchsifat.azurecr.io           # your login server from step 2
docker build --platform linux/amd64 -t $REG/visual-search:v1 .
```

The first build takes about 10–20 minutes, mostly downloading PyTorch. Test it locally
before uploading:

```bash
docker run --rm -p 8000:8000 $REG/visual-search:v1
# open http://localhost:8000, search something, then Ctrl+C
```

Push it to Azure. Use the username and password from step 2 when asked. At your upload
speed, expect about 30–60 minutes:

```bash
docker login $REG
docker push $REG/visual-search:v1
```

Check it arrived: Portal → your registry → **Services → Repositories** → `visual-search` → tag `v1`.

## Step 4: Create the Kubernetes cluster (Portal)

1. Search **"Kubernetes services"** → **+ Create** → **Kubernetes cluster**.
   Don't pick "Automatic": it needs more vCPUs than your trial allows.
2. **Basics** tab:
   - Resource group: **visual-search-rg**
   - Cluster preset configuration: **Dev/Test**
   - Kubernetes cluster name: **visual-search-aks**
   - Region: **East US**
   - Availability zones: **None**
   - AKS pricing tier: **Free**
   - Leave the Kubernetes version as the default.
3. **Node pools** tab → click **agentpool** to edit it:
   - Node size → **Choose a size** → **Standard_D2s_v7** (2 vCPUs, 8 GB).
     The v5 sizes aren't offered to free-trial subscriptions; D2as_v7 or D2s_v4 also work.
     Avoid the 4 GB sizes (`D2ls_v7`, `D2als_v7`).
   - Scale method: **Manual**, Node count: **1**
   - **Update**
4. **Networking** tab: leave the defaults.
5. **Integrations** tab → **Container registry**: select **visualsearchsifat**.
   This lets the cluster pull your private image without passwords.
6. **Monitoring** tab: turn off Container insights, Managed Prometheus and alerts, to
   save cost and vCPUs.
7. **Review + create** → **Create**. It takes about 5–10 minutes.

If it fails with a **quota** error, it's the 4-vCPU limit. Stop the other cluster
(`spam-email-validation-aks` → Overview → **Stop**) and try again.

## Step 5: Deploy the app (Portal)

1. Open **visual-search-aks** → left menu **Kubernetes resources → Workloads**.
2. **+ Create** → **Apply a YAML**.
3. Paste the contents of [deploy/k8s/visual-search.yaml](../deploy/k8s/visual-search.yaml).
   Replace `<REGISTRY>` with your registry name, so the line reads
   `image: visualsearchsifat.azurecr.io/visual-search:v1`.
4. **Add**.

## Step 6: HTTPS on a free Azure hostname

The Service in step 5 is internal (`ClusterIP`). Public traffic enters through an NGINX
ingress controller, which also terminates HTTPS with a free Let's Encrypt certificate.
Installing the controller and cert-manager needs Helm. Run these either from your Mac
(after `az aks get-credentials -n visual-search-aks -g visual-search-rg`) or in the Portal
under **visual-search-aks → Run command**:

```bash
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm repo add jetstack https://charts.jetstack.io
helm repo update

# Public front door, with a free DNS name: <label>.<region>.cloudapp.azure.com
helm install ingress-nginx ingress-nginx/ingress-nginx -n ingress-nginx --create-namespace \
  --set controller.service.annotations."service\.beta\.kubernetes\.io/azure-dns-label-name"=visual-search-sifat \
  --set controller.service.annotations."service\.beta\.kubernetes\.io/azure-load-balancer-health-probe-request-path"=/healthz

# Issues and renews certificates automatically
helm install cert-manager jetstack/cert-manager -n cert-manager --create-namespace --set crds.enabled=true
```

Then **Apply a YAML** with [https/cluster-issuer.yaml](../deploy/k8s/https/cluster-issuer.yaml)
and [https/ingress.yaml](../deploy/k8s/https/ingress.yaml). The hostname's region must be
the **cluster's** region (this cluster is in `eastus2`). The certificate is issued in under a minute.

Open **https://visual-search-sifat.eastus2.cloudapp.azure.com**. HTTP redirects to HTTPS.

**Check the pod:** **Workloads → Pods** → `visual-search-…` should show Status *Running*
and Ready *1/1*. Click it → **Live logs** to see the server output.

| Symptom | Likely cause |
|---|---|
| Pod stuck in `ImagePullBackOff` | Registry not attached (step 4.5), or a typo in the image name |
| `exec format error` in the logs | The image was built for ARM; rebuild with `--platform linux/amd64` |
| Pod `Running` but Ready `0/1` for a while | Still loading the model; the readiness check waits for `/health` |
| `OOMKilled` | Not enough memory; use a node with 8 GB |
| Browser says "not private" | Certificate not issued yet: `kubectl describe certificate visual-search-tls` |
| Upload fails with 413 | The Ingress is missing the `proxy-body-size` annotation |

## Step 7: Ship a new version

Rebuild with a new tag (`v2`), push it, then **Workloads → Deployments → visual-search →
YAML** → change `:v1` to `:v2` → **Review + save**. Kubernetes starts the new pod, waits
until it's ready, then removes the old one, so there's no downtime.

## Costs and cleanup

- The node VM costs about $70 per month if left running, paid from the trial credit.
  The Basic registry costs about $5 per month. The load balancer and public IP cost a little.
- **Pause:** AKS → Overview → **Stop**. Compute stops billing, and **Start** brings it back.
- **Delete everything:** Resource groups → **visual-search-rg** → **Delete resource group**.
  This also deletes the Azure ML workspace in the same group.

## Known limitations

- **Uploads aren't persistent.** "Add to index" writes to the pod's disk, so restarts lose
  it. The fix is an Azure Files volume, or storing the index in a database.
- **One replica.** For the same reason, more than one pod would serve different indexes.
- **Certificates renew automatically** (cert-manager, every ~60 days) as long as the cluster is running.
