# 按分类浅克隆 Tencent-Hunyuan 仓库到本地
#   用法（在仓库根目录 D:\WorkSpace\workspace\Hunyuan 下执行）：
#       pwsh -File clone-repos.ps1 -Category agent
#       pwsh -File clone-repos.ps1 -Category infer
#       pwsh -File clone-repos.ps1 -Category agent,infer
#       pwsh -File clone-repos.ps1 -Category infer -Depth 1 -SkipLFS   # 默认即浅克隆
#
# 说明：
#   - 默认 --depth 1 浅克隆（只取最新快照，不含 Git 历史），显著加快下载、减小占用
#   - 默认 GIT_LFS_SKIP_SMUDGE=1，跳过 LFS 大文件（模型权重/演示视频），需要时再单独 git lfs pull
#   - 已存在的目录会跳过（幂等）；加 -Pull 可改为在原地 git pull 更新
#   - 完整历史请用 -FullHistory

[CmdletBinding()]
param(
    [ValidateSet('agent','infer','out-of-scope')]
    [string[]]$Category = @('agent','infer'),

    [int]$Depth = 1,
    [switch]$FullHistory,
    [switch]$Pull,
    [switch]$SkipLFS = $true,
    [string]$Root = $PSScriptRoot
)

$ErrorActionPreference = 'Continue'
if (-not $Root) { $Root = (Get-Location).Path }

# 仓库 → 分类映射（与 _research/repos-classification.csv 保持一致，共 85 个）
$manifest = @{
    'agent' = @(
        'UniRL','C3-Benchmark','VisualNeedle','RepoProbe','ExplorationBench',
        'PlanningBench','ArtifactsBenchmark','Hyra-results',
        'HY-Embodied','HY-Embodied-0.5-X','Hy-Embodied-0.5-VLA',
        'Hy-Embodied-RxBrain-1.0','Hy-Embodied-RoboFusion','Hunyuan3D-WorldClaw'
    )
    'infer' = @(
        # 推理优化 / 高效注意力库
        'HiLS-Attention','flex-block-attn','Prism','Rosetta-inference','AuK',
        'Simple-Attention-Sparsification','FlashVDM','DisCa',
        # LLM 与翻译模型
        'Hy3','Hy3-preview','Hy4-preview','Hy-MT','Hy-MT2','Hunyuan-MT',
        'Hunyuan-A13B','Hunyuan-7B','Hunyuan-4B','Hunyuan-1.8B','Hunyuan-0.5B',
        'Tencent-Hunyuan-7B-0124','Tencent-Hunyuan-Large','Hunyuan-TurboS','Hunyuan-TurboS3',
        # 多模态理解
        'HunyuanOCR','HunyuanVision',
        # 图像生成 / 编辑
        'HunyuanImage-3.0','HunyuanImage-2.1','HunyuanDiT','iFSQ','HY-WU',
        # 视频生成
        'HunyuanVideo','HunyuanVideo-1.5','HunyuanVideo-I2V','HunyuanVideo-Avatar',
        'HunyuanVideo-Foley','HunyuanCustom','Hunyuan-GameCraft-1.0',
        'HunyuanPortrait','InstantCharacter','OmniWeaving',
        # 3D 生成
        'Hunyuan3D-1','Hunyuan3D-2','Hunyuan3D-2.1','Hunyuan3D-Omni',
        'Hunyuan3D-Part','Hunyuan3D-Buffalo1.0','bpt',
        # 世界模型
        'HY-World-2.0','HunyuanWorld-1.0','HunyuanWorld-Voyager','HunyuanWorld-Mirror','HY-WorldPlay',
        # 其他生成
        'HY-Motion-1.0','R-DMesh','UniCom','HY-SOAR'
    )
    'out-of-scope' = @(
        'GEAR','Precise','MixGRPO','SRPO','SAGE-GRPO','HY-Video-PRFL','GradLoc',
        'DRIVE-RLVR','Thinking-Free_Policy_Initialization','UnifiedReward','Twins',
        'AutoCodeBenchmark','CL-bench','HY3D-Bench','evolve-scaler'
    )
}

$org = 'Tencent-Hunyuan'
$targets = foreach ($c in $Category) {
    foreach ($r in $manifest[$c]) { [pscustomobject]@{ Repo = $r; Category = $c } }
}
$targets = $targets | Sort-Object Repo -Unique

Write-Host "==> 目标根目录 : $Root"
Write-Host "==> 分类        : $($Category -join ', ')"
Write-Host "==> 仓库数量    : $($targets.Count)"
Write-Host "==> 克隆模式    : $(if ($FullHistory) { '完整历史' } else { "浅克隆 --depth $Depth" })$(if ($SkipLFS -and -not $FullHistory) { ' + 跳过 LFS' } else { '' })"
Write-Host ''

$ok = 0; $skip = 0; $fail = @()
$sw = [System.Diagnostics.Stopwatch]::StartNew()

foreach ($t in $targets) {
    $dest = Join-Path (Join-Path $Root $t.Category) $t.Repo
    $url  = "https://github.com/$org/$($t.Repo).git"

    if (Test-Path (Join-Path $dest '.git')) {
        if ($Pull) {
            Write-Host ("[PULL] {0,-34} -> {1}" -f $t.Repo, $t.Category)
            git -C $dest pull --ff-only 2>&1 | ForEach-Object { "       $_" }
            if ($LASTEXITCODE -eq 0) { $ok++ } else { $fail += $t.Repo }
        } else {
            Write-Host ("[SKIP] {0,-34} 已存在" -f $t.Repo); $skip++
        }
        continue
    }

    New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null

    $gitArgs = @('clone', '--quiet')
    if (-not $FullHistory) { $gitArgs += @('--depth', $Depth) }
    $gitArgs += @($url, $dest)

    Write-Host ("[GET ] {0,-34} -> {1}" -f $t.Repo, $t.Category)
    if ($SkipLFS -and -not $FullHistory) { $env:GIT_LFS_SKIP_SMUDGE = '1' }
    & git @gitArgs 2>&1 | ForEach-Object { "       $_" }
    Remove-Item Env:\GIT_LFS_SKIP_SMUDGE -ErrorAction SilentlyContinue

    if ($LASTEXITCODE -eq 0) { $ok++ } else { Write-Host ("       !! 失败: {0}" -f $t.Repo); $fail += $t.Repo }
}

$sw.Stop()
Write-Host ''
Write-Host ('=' * 62)
Write-Host ("完成: 成功 {0} / 跳过 {1} / 失败 {2}  ({3:mm\:ss})" -f $ok, $skip, $fail.Count, $sw.Elapsed)
if ($fail.Count) { Write-Host ("失败列表: {0}" -f ($fail -join ', ')); exit 1 }
