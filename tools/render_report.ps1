# Native Word page images avoid the unavailable bundled LibreOffice on Windows.
# This is an optional report-layout check, not part of training or Streamlit.
$ErrorActionPreference = 'Stop'
$reportWorkspace = Split-Path $PSScriptRoot -Parent
$reportSource = Join-Path $reportWorkspace 'submission/COS30082_Report.docx'
$reportQaFolder = Join-Path $reportWorkspace 'tmp/report_qa'
New-Item -ItemType Directory -Path $reportQaFolder -Force | Out-Null
$reportApp = $null
$reportDocument = $null
try {
    $reportApp = New-Object -ComObject Word.Application
    $reportApp.Visible = $false
    $reportApp.DisplayAlerts = 0
    $reportDocument = $reportApp.Documents.Open($reportSource, $false, $true)
    $reportDocument.ActiveWindow.View.Type = 3
    $reportDocument.Repaginate()
    $reportPages = $reportDocument.ActiveWindow.Panes.Item(1).Pages
    Write-Output ('Report page count: ' + $reportPages.Count)
    if ($reportPages.Count -gt 4) { throw 'Report exceeds four pages' }
    for ($reportPageIndex=1; $reportPageIndex -le $reportPages.Count; $reportPageIndex++) {
        [byte[]]$reportPageBytes = $reportPages.Item($reportPageIndex).EnhMetaFileBits
        $reportPagePath = Join-Path $reportQaFolder ('page-' + $reportPageIndex + '.emf')
        [System.IO.File]::WriteAllBytes($reportPagePath, $reportPageBytes)
        Write-Output $reportPagePath
    }
} finally {
    if ($null -ne $reportDocument) { $reportDocument.Close(0) }
    if ($null -ne $reportApp) { $reportApp.Quit() }
}
# Convert the page EMFs to PNGs with the artifact runtime Python and Pillow.
# Image.open(path).load(); image.save(path.with_suffix('.png'))
