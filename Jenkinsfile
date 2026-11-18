pipeline {
    agent any

    environment {
        // ── Identity ───────────────────────────────────────────────────────
        STUDENT_ID     = 'SP23-BAI-031'
        APP_NAME       = 'devops-labmidterm-sp23-bai-031'

        // ── Docker ─────────────────────────────────────────────────────────
        DOCKER_IMAGE   = "faiqster/${APP_NAME}"
        DOCKER_TAG     = "${BUILD_NUMBER}"

        // ── AWS EC2 ────────────────────────────────────────────────────────
        EC2_HOST       = '3.110.121.46'
        EC2_USER       = 'ubuntu'
        CONTAINER_PORT = '8000'
        HOST_PORT      = '8000'
        CONTAINER_NAME = "${APP_NAME}"

        // ── Jenkins credential IDs
        SSH_CRED_ID    = 'ec2-pem-key'
        DOCKER_CRED_ID = 'dockerhub-creds'
    }

    options {
        timestamps()
        timeout(time: 30, unit: 'MINUTES')
        disableConcurrentBuilds()
    }

    stages {

        // ── 1. Checkout ────────────────────────────────────────────────────
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    env.GIT_COMMIT_SHORT = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
                    env.GIT_COMMIT       = sh(script: 'git rev-parse HEAD',         returnStdout: true).trim()
                }
                echo "Building commit ${env.GIT_COMMIT_SHORT} for ${STUDENT_ID}"
            }
        }

        // ── 2. Set up Python environment ───────────────────────────────────
        stage('Setup Python') {
            steps {
                sh '''
                    python3 -m venv .venv
                    . .venv/bin/activate
                    pip install --upgrade pip --quiet
                    pip install -r requirements.txt --quiet
                    pip install pytest --quiet
                '''
            }
        }

        // ── 3. Generate / refresh dataset ─────────────────────────────────
        stage('Fetch Latest Data') {
            steps {
                sh '''
                    . .venv/bin/activate
                    mkdir -p dataset
                    python generate_dataset.py --rows 300
                    echo "Dataset ready: $(wc -l < dataset/train.csv) lines"
                '''
            }
        }

        // ── 4. Train ───────────────────────────────────────────────────────
        stage('Train Model') {
            environment {
                BUILD_COMMIT = "${env.GIT_COMMIT}"
            }
            steps {
                sh '''
                    . .venv/bin/activate
                    python train.py
                    echo "--- Metrics ---"
                    cat artifacts/metrics.json
                '''
            }
        }

        // ── 5. Tests ───────────────────────────────────────────────────────
        stage('Run Tests') {
            steps {
                sh '''
                    . .venv/bin/activate
                    if [ -d tests ]; then
                        python -m pytest tests/ -v --tb=short
                    else
                        echo "No tests/ directory found – running inline smoke tests"
                        python - <<'PYEOF'
import json, joblib
from pathlib import Path

metrics = json.loads(Path("artifacts/metrics.json").read_text())
assert metrics["student_id"] == "SP23-BAI-031", "Wrong student ID in metrics"
assert 0.0 <= metrics["accuracy"] <= 1.0,        "Accuracy out of range"
assert metrics["training_rows"] >= 100,           "Too few training rows"

bundle = joblib.load("artifacts/model.joblib")
assert "model" in bundle and "features" in bundle, "Bundle missing keys"
assert len(bundle["features"]) == 4,              "Expected 4 feature columns"

import pandas as pd
sample = pd.DataFrame([{
    "basket_volatility": 0.5,
    "stock_gap": -0.2,
    "promotion_load": 0.8,
    "delivery_delay": 0.1
}])
proba = bundle["model"].predict_proba(sample)
assert proba.shape == (1, 2), "predict_proba shape wrong"
print("All smoke tests passed. Accuracy:", metrics["accuracy"])
PYEOF
                    fi
                '''
            }
        }

        // ── 6. Build Docker image ──────────────────────────────────────────
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build("${DOCKER_IMAGE}:${DOCKER_TAG}")
                    docker.build("${DOCKER_IMAGE}:latest")
                }
            }
        }

        // ── 7. Push to Docker Hub ──────────────────────────────────────────
        stage('Push Docker Image') {
            steps {
                script {
                    try {
                        docker.withRegistry('https://index.docker.io/v1/', DOCKER_CRED_ID) {
                            def img = docker.image("${DOCKER_IMAGE}:${DOCKER_TAG}")
                            img.push()
                            img.push('latest')
                        }
                        echo "Successfully pushed ${DOCKER_IMAGE}:${DOCKER_TAG} and :latest to Docker Hub"
                    } catch (Exception e) {
                        echo "Docker Hub push skipped or failed: ${e.message}"
                        echo "Will deploy using locally built Docker image."
                    }
                }
            }
        }

        // ── 8. Deploy to AWS EC2 ───────────────────────────────────────────
        stage('Deploy to EC2') {
            steps {
                sshagent(credentials: [SSH_CRED_ID]) {
                    sh """
                        ssh -o StrictHostKeyChecking=no ${EC2_USER}@${EC2_HOST} '
                            set -e

                            # Pull latest image if available in registry, else use local image
                            docker pull ${DOCKER_IMAGE}:latest 2>/dev/null || echo "Using locally built image ${DOCKER_IMAGE}:latest"

                            # Stop & remove old container if exists
                            docker stop  ${CONTAINER_NAME} 2>/dev/null || true
                            docker rm    ${CONTAINER_NAME} 2>/dev/null || true

                            # Start new container
                            docker run -d \\
                                --name  ${CONTAINER_NAME} \\
                                --restart always \\
                                -p ${HOST_PORT}:${CONTAINER_PORT} \\
                                ${DOCKER_IMAGE}:latest

                            # Prune untagged dangling images
                            docker image prune -f
                        '
                    """
                }
            }
        }

        // ── 9. Health-check ────────────────────────────────────────────────
        stage('Health Check') {
            steps {
                sh """
                    echo "Waiting 15 s for container to start..."
                    sleep 15
                    STATUS=\$(curl -s -o /dev/null -w '%{http_code}' http://${EC2_HOST}:${HOST_PORT}/health || echo 000)
                    if [ "\$STATUS" = "200" ]; then
                        echo "Health check passed (HTTP \$STATUS)"
                    else
                        echo "Health check FAILED (HTTP \$STATUS)"
                        exit 1
                    fi
                """
            }
        }
    }

    // ── Post-build notifications ───────────────────────────────────────────
    post {
        success {
            echo """
╔══════════════════════════════════════════════╗
║  DEPLOYMENT SUCCESSFUL                       ║
║  Student : ${STUDENT_ID}                     ║
║  Build   : #${BUILD_NUMBER}                  ║
║  Commit  : ${env.GIT_COMMIT_SHORT}           ║
║  App URL : http://${EC2_HOST}:${HOST_PORT}   ║
╚══════════════════════════════════════════════╝"""
        }
        failure {
            echo "Pipeline FAILED on stage. Check logs above."
        }
        always {
            sh 'rm -rf .venv || true'
        }
    }
}
