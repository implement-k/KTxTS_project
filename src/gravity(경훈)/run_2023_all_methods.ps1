$ErrorActionPreference = "Stop"

$methods = @(
    "lgbm",
    "trip_rate",
    "cross_class",
    "linear_regression"
)

foreach ($method in $methods) {
    Write-Host "=== Gravity 2023-only / $method / test ==="
    python .\train_and_val.py `
        --model_type $method `
        --imputation zero `
        --train_years 2023 `
        --eval_years 2023 `
        --eval_splits test
}
