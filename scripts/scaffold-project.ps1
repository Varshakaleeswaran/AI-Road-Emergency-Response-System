
# Run this script from the root of AI-Road-Emergency-Response-System.
# It creates folders and safe placeholder files only. It never overwrites files.

$projectDirs = @(
    '.github/workflows',
    '.github/ISSUE_TEMPLATE',
    'ai/configs',
    'ai/data/raw',
    'ai/data/processed',
    'ai/data/sample',
    'ai/models',
    'ai/notebooks',
    'ai/scripts',
    'ai/src/detection',
    'ai/src/preprocessing',
    'ai/src/evaluation',
    'ai/src/schemas',
    'ai/src/services',
    'ai/src/utils',
    'ai/tests',
    'backend/alembic',
    'backend/app/api/v1/endpoints',
    'backend/app/core',
    'backend/app/db',
    'backend/app/models',
    'backend/app/schemas',
    'backend/app/services',
    'backend/app/repositories',
    'backend/app/integrations',
    'backend/app/websocket',
    'backend/app/tasks',
    'backend/tests/unit',
    'backend/tests/integration',
    'backend/tests/api',
    'frontend/public',
    'frontend/src/app',
    'frontend/src/components/common',
    'frontend/src/components/layout',
    'frontend/src/components/map',
    'frontend/src/features/auth',
    'frontend/src/features/dashboard',
    'frontend/src/features/incidents',
    'frontend/src/features/ai-detection',
    'frontend/src/features/ambulances',
    'frontend/src/features/hospitals',
    'frontend/src/features/departments',
    'frontend/src/features/notifications',
    'frontend/src/features/audit-logs',
    'frontend/src/features/simulation',
    'frontend/src/pages',
    'frontend/src/services',
    'frontend/src/hooks',
    'frontend/src/stores',
    'frontend/src/types',
    'frontend/src/utils',
    'frontend/src/styles',
    'frontend/tests',
    'database/schema',
    'database/seed',
    'database/scripts',
    'simulation/scenarios',
    'simulation/fixtures',
    'simulation/scripts',
    'docs/adr',
    'infra/docker',
    'infra/nginx',
    'scripts',
    'tests/end-to-end',
    'tests/performance',
    'tests/security'
)

foreach ($projectDir in $projectDirs) {
    if (-not (Test-Path -LiteralPath $projectDir)) {
        New-Item -ItemType Directory -Path $projectDir -Force | Out-Null
    }

    $keepFile = Join-Path $projectDir '.gitkeep'
    if (-not (Test-Path -LiteralPath $keepFile)) {
        New-Item -ItemType File -Path $keepFile -Force | Out-Null
    }
}

$emptyFiles = @(
    'ai/requirements.txt',
    'ai/README.md',
    'ai/src/__init__.py',
    'ai/src/detection/__init__.py',
    'ai/src/preprocessing/__init__.py',
    'ai/src/evaluation/__init__.py',
    'ai/src/schemas/__init__.py',
    'ai/src/services/__init__.py',
    'ai/src/utils/__init__.py',
    'backend/requirements.txt',
    'backend/README.md',
    'backend/.env.example',
    'backend/app/__init__.py',
    'backend/app/api/__init__.py',
    'backend/app/api/v1/__init__.py',
    'backend/app/api/v1/endpoints/__init__.py',
    'backend/app/core/__init__.py',
    'backend/app/db/__init__.py',
    'backend/app/models/__init__.py',
    'backend/app/schemas/__init__.py',
    'backend/app/services/__init__.py',
    'backend/app/repositories/__init__.py',
    'backend/app/integrations/__init__.py',
    'backend/app/websocket/__init__.py',
    'backend/app/tasks/__init__.py',
    'frontend/.env.example',
    'frontend/README.md',
    'database/schema/er-diagram.md',
    'database/schema/initial-schema.sql',
    'database/seed/ambulances.json',
    'database/seed/hospitals.json',
    'database/seed/departments.json',
    'simulation/scenarios/normal-accident.json',
    'simulation/scenarios/false-ai-detection.json',
    'simulation/scenarios/witness-report.json',
    'simulation/scenarios/duplicate-report.json',
    'simulation/scenarios/multiple-casualties.json',
    'simulation/scenarios/ambulance-unavailable.json',
    'simulation/scenarios/hospital-unavailable.json',
    'simulation/scenarios/network-interruption.json',
    'docs/requirements.md',
    'docs/architecture.md',
    'docs/scope.md',
    'docs/incident-lifecycle.md',
    'docs/api.md',
    'docs/database.md',
    'docs/testing.md',
    'docs/security.md',
    'docs/deployment.md',
    'CONTRIBUTING.md'
)

foreach ($emptyFile in $emptyFiles) {
    if (-not (Test-Path -LiteralPath $emptyFile)) {
        New-Item -ItemType File -Path $emptyFile -Force | Out-Null
    }
}

if (-not (Test-Path -LiteralPath '.gitignore')) {
    @(
        '# Python',
        '.venv/',
        '__pycache__/',
        '*.py[cod]',
        '.pytest_cache/',
        '',
        '# Node / React',
        'node_modules/',
        'dist/',
        '',
        '# Environment files',
        '.env',
        '.env.*',
        '!.env.example',
        '',
        '# AI data and trained models',
        'ai/data/raw/',
        'ai/data/processed/',
        'ai/models/',
        '',
        '# Tooling and operating system files',
        '.vscode/*.local.json',
        '.DS_Store',
        'Thumbs.db'
    ) | Set-Content -LiteralPath '.gitignore' -Encoding utf8
}

if (-not (Test-Path -LiteralPath '.env.example')) {
    '# Shared non-secret configuration belongs here. Never commit a real .env file.' |
        Set-Content -LiteralPath '.env.example' -Encoding utf8
}

Write-Host 'Project scaffold created. Review it, then run git status.'

