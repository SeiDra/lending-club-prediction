# ☁️ Guide de Déploiement AWS — Credit Risk Analyzer

> **Projet MLOps** | LendingClub Default Prediction | DU Sorbonne Data Analytics  
> Stack : GitHub Actions → Docker Hub → Amazon ECR → Amazon ECS (Fargate)

---

## Architecture de déploiement

```
Git Push (main)
     │
     ▼
GitHub Actions ──► CI : black → pylint → pytest
     │
     ▼ (si CI passe)
     ├──► Docker build + push → Docker Hub  (github-docker-cicd.yaml)
     │
     └──► ECR push + ECS deploy             (aws.yml)
                                │
                                ▼
                    Application Load Balancer
                    http://lending-club-alb-xxxx.eu-west-3.elb.amazonaws.com
```

---

## Prérequis

- Compte AWS actif (région : `eu-west-3` — Paris)
- AWS CLI installé et configuré localement
- Docker Desktop installé
- Compte Docker Hub avec un repo `lending-club-mlops`
- Repo GitHub : `SeiDra/lending-club-prediction`

---

## Étape 01 — Utilisateur IAM

> **Rôle** : Définir qui a le droit d'agir sur AWS depuis GitHub Actions.

1. Aller dans **IAM → Utilisateurs → Créer un utilisateur**
2. Nom : `github-actions-mlops`
3. Attacher les politiques suivantes :
   - `AmazonEC2ContainerRegistryFullAccess`
   - `AmazonECS_FullAccess`
4. Aller dans **Informations d'identification de sécurité → Créer une clé d'accès**
   - Type : *Application s'exécutant en dehors d'AWS*
5. Copier `AWS_ACCESS_KEY_ID` et `AWS_SECRET_ACCESS_KEY`
6. Les ajouter dans **GitHub → Settings → Secrets → Actions** :
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`

---

## Étape 02 — Amazon ECR (Elastic Container Registry)

> **Rôle** : Stocker les images Docker prêtes à déployer.

```bash
# Créer le dépôt ECR via AWS CLI
aws ecr create-repository \
    --repository-name seidra/lending-club-mlops \
    --region eu-west-3
```

Ou via la console AWS :
1. **ECR → Dépôts → Créer un dépôt**
2. Visibilité : **Privé**
3. Nom : `seidra/lending-club-mlops`
4. Région : `eu-west-3`

> 💡 L'URI du dépôt sera de la forme :  
> `XXXXXXXXXXXX.dkr.ecr.eu-west-3.amazonaws.com/seidra/lending-club-mlops`

---

## Étape 03 — ECS Cluster

> **Rôle** : Regrouper les services dans un même espace logique AWS.

```bash
aws ecs create-cluster \
    --cluster-name LendingClubCluster \
    --region eu-west-3
```

Ou via la console :
1. **ECS → Clusters → Créer un cluster**
2. Nom : `LendingClubCluster`
3. Infrastructure : **AWS Fargate** (serverless — pas de serveur à gérer)
4. Région : `eu-west-3`

---

## Étape 04 — Task Definition (Définition de tâche)

> **Rôle** : Décrire comment lancer le container (ressources, port, image).

1. **ECS → Définitions de tâches → Créer une nouvelle définition**
2. Nom de la famille : `lending-club-task`
3. Infrastructure : **AWS Fargate**
4. CPU : `0.5 vCPU` | Mémoire : `1 Go` *(suffisant pour Streamlit)*
5. **Ajouter un conteneur** :

| Champ | Valeur |
|---|---|
| Nom du conteneur | `credit-app-mlops` |
| Image URI | `XXXX.dkr.ecr.eu-west-3.amazonaws.com/seidra/lending-club-mlops:latest` |
| Port | `8501` (TCP) |
| Protocole de journalisation | `awslogs` |

6. Variables d'environnement *(optionnel)* :
   - Aucune requise pour cette app

7. Cliquer **Créer**

---

## Étape 05 — Security Groups (Groupes de sécurité)

> **Rôle** : Contrôler quel trafic réseau peut entrer et sortir.

1. **EC2 → Groupes de sécurité → Créer un groupe de sécurité**
2. Nom : `lending-club-sg`
3. VPC : VPC par défaut
4. **Règles entrantes** :

| Type | Protocole | Port | Source |
|---|---|---|---|
| HTTP | TCP | 80 | `0.0.0.0/0` |
| Custom TCP | TCP | 8501 | `0.0.0.0/0` |

5. **Règles sortantes** : Tout le trafic (par défaut)

---

## Étape 06 — ECS Service

> **Rôle** : Garantir qu'un container tourne en permanence.

1. **ECS → Clusters → LendingClubCluster → Créer un service**
2. Environnement : **Fargate**
3. Définition de tâche : `lending-club-task` (dernière révision)
4. Nom du service : `lending-club-service`
5. Nombre de tâches souhaitées : `1`
6. **Mise en réseau** :
   - VPC : VPC par défaut
   - Sous-réseaux : sélectionner au moins 2 sous-réseaux
   - Groupe de sécurité : `lending-club-sg`
   - IP publique : **Activée**
7. **Load balancing** : Sélectionner l'ALB créé à l'étape 07
8. Cliquer **Créer**

---

## Étape 07 — Application Load Balancer (ALB)

> **Rôle** : Donner une URL fixe et permanente à l'application.

1. **EC2 → Load Balancers → Créer un load balancer**
2. Type : **Application Load Balancer**
3. Nom : `lending-club-alb`
4. Schéma : **Orienté vers Internet**
5. Zones de disponibilité : sélectionner 2 sous-réseaux publics
6. Groupe de sécurité : `lending-club-sg`
7. **Écouteurs** :
   - Protocole : HTTP | Port : 80
8. **Groupe cible** :
   - Créer un nouveau groupe cible
   - Type : **IP**
   - Nom : `lending-club-tg`
   - Port : `8501`
   - Protocole : HTTP
   - Vérification d'état : chemin `/` 

> 💡 L'URL de l'app sera :  
> `http://lending-club-alb-XXXXXXXXXX.eu-west-3.elb.amazonaws.com`

---

## Secrets GitHub à configurer

Aller dans **GitHub → Settings → Secrets and variables → Actions → New repository secret** :

| Secret | Description | Où le trouver |
|---|---|---|
| `AWS_ACCESS_KEY_ID` | Clé d'accès IAM | IAM → Utilisateurs → Clés d'accès |
| `AWS_SECRET_ACCESS_KEY` | Clé secrète IAM | IAM → Utilisateurs → Clés d'accès |
| `DOCKER_USER` | Identifiant Docker Hub | docker.com → Account |
| `DOCKER_PASSWORD` | Token Docker Hub | Docker Hub → Account Settings → Security |
| `REPO_NAME` | Nom du repo Docker Hub | `lending-club-mlops` |

---

## Variables à mettre à jour dans `aws.yml`

Après avoir créé les ressources AWS, mettre à jour les variables `env` dans `.github/workflows/aws.yml` :

```yaml
env:
  AWS_REGION: eu-west-3
  ECR_REPOSITORY: seidra/lending-club-mlops   # nom du dépôt ECR
  ECS_SERVICE: lending-club-service            # nom du service ECS
  ECS_CLUSTER: LendingClubCluster              # nom du cluster ECS
  ECS_TASK_DEFINITION: lending-club-task       # nom de la task definition
  CONTAINER_NAME: credit-app-mlops             # nom du conteneur dans la task def
```

---

## Vérification du déploiement

```bash
# Vérifier que le service ECS tourne
aws ecs describe-services \
    --cluster LendingClubCluster \
    --services lending-club-service \
    --region eu-west-3 \
    --query "services[0].{Status:status,Running:runningCount,Desired:desiredCount}"

# Récupérer l'URL du Load Balancer
aws elbv2 describe-load-balancers \
    --names lending-club-alb \
    --region eu-west-3 \
    --query "LoadBalancers[0].DNSName" \
    --output text
```

---

## Coûts estimés (région eu-west-3)

| Service | Coût estimé |
|---|---|
| Fargate (0.5 vCPU / 1 Go RAM, 24h/7j) | ~15 $/mois |
| ECR (stockage image ~500 Mo) | ~0.05 $/mois |
| ALB | ~18 $/mois |
| **Total estimé** | **~33 $/mois** |

> ⚠️ Penser à **stopper le service ECS** après la présentation pour éviter des frais inutiles :
> ```bash
> aws ecs update-service \
>     --cluster LendingClubCluster \
>     --service lending-club-service \
>     --desired-count 0 \
>     --region eu-west-3
> ```

---

*Document généré dans le cadre du projet MLOps — DU Sorbonne Data Analytics 2026*
