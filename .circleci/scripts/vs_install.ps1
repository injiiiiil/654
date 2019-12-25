$VS_DOWNLOAD_LINK = "https://aka.ms/vs/15/release/vs_buildtools.exe"
$VS_INSTALL_ARGS= @("--nocache","--quiet","--wait", "--add Microsoft.VisualStudio.Workload.VCTools",
                                                    "--add Microsoft.VisualStudio.Component.VC.Tools.14.11",
                                                    "--add Microsoft.Component.MSBuild",
                                                    "--add Microsoft.VisualStudio.Component.Roslyn.Compiler",
                                                    "--add Microsoft.VisualStudio.Component.TextTemplating",
                                                    "--add Microsoft.VisualStudio.Component.VC.CoreIde",
                                                    "--add Microsoft.VisualStudio.Component.VC.Redist.14.Latest",
                                                    "--add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Core",
                                                    "--add Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                                                    "--add Microsoft.VisualStudio.Component.VC.Tools.14.11",
                                                    "--add Microsoft.VisualStudio.ComponentGroup.NativeDesktop.Win81")

curl.exe -kL $VS_DOWNLOAD_LINK --output vs_installer.exe
if ($LASTEXITCODE -ne 0) {
    exit 1
}

$exitCode = Start-Process "${PWD}\vs_installer.exe" -ArgumentList $VS_INSTALL_ARGS -NoNewWindow -Wait -PassThru
Remove-Item -Path vs_installer.exe -Force
if (($exitCode -ne 0) -and ($exitCode -ne 3010)) {
    exit 1
}
