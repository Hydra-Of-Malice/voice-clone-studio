; Inno Setup script for Voice Clone Studio (Windows 10/11 x64). Build with scripts\build-installer.ps1.
;
; Self-contained: ships the relocatable Python 3.11 runtime with the CUDA 12.6 stack ({app}\runtime) and every
; model ({app}\models), so nothing is downloaded on the user's PC. User data (voice profiles, generated audio,
; logs) lives in %USERPROFILE%\.voice-clone and is never touched by the uninstaller.

#if VER < EncodeVer(6,3,0,0)
  #error This script needs Inno Setup 6.3 or newer
#endif
#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef OutputDir
  #define OutputDir "..\build\installer"
#endif
#ifndef RuntimeDir
  #define RuntimeDir "..\build\runtime"
#endif
#ifndef ModelsDir
  #define ModelsDir "..\build\models"
#endif
#define AppName "Voice Clone Studio"
#define AppURL "https://github.com/Hydra-Of-Malice/voice-clone-studio"
#define SrcDir ".."
#define PythonW "{app}\runtime\pythonw.exe"
#define PythonExe "{app}\runtime\python.exe"

[Setup]
AppId={{B4D1F6A2-3E57-4C1B-9A60-7D2E5F8C1A34}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=Voice Clone Studio
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
; Per-user install (no administrator rights): {app} = %LOCALAPPDATA%\Programs\Voice Clone Studio
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0.17763
OutputDir={#OutputDir}
OutputBaseFilename=VoiceCloneStudio-Setup-{#AppVersion}
#ifexist "app.ico"
SetupIconFile=app.ico
UninstallDisplayIcon={app}\installer\app.ico
#endif
UninstallDisplayName={#AppName}
SolidCompression=no
; Mostly incompressible payload (CUDA DLLs, model weights): fast deflate keeps the build to minutes
Compression=zip/1
; Slices under 2 GB: the per-file limit of GitHub Releases. All files must sit in one folder when installing.
DiskSpanning=yes
DiskSliceSize=2000000000
WizardStyle=modern
UsePreviousAppDir=yes
InfoBeforeFile=before-install.txt

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SrcDir}\vc\*"; DestDir: "{app}\vc"; Excludes: "__pycache__,*.pyc,*.pyo"; Flags: recursesubdirs ignoreversion
Source: "{#SrcDir}\web\dist\*"; DestDir: "{app}\web\dist"; Flags: recursesubdirs ignoreversion
Source: "{#SrcDir}\docs\USER-GUIDE.md"; DestDir: "{app}"; DestName: "User guide.txt"; Flags: ignoreversion isreadme
Source: "{#SrcDir}\THIRD-PARTY-NOTICES.md"; DestDir: "{app}"; DestName: "Third-party notices.txt"; Flags: ignoreversion
#ifexist "app.ico"
Source: "app.ico"; DestDir: "{app}\installer"; Flags: ignoreversion
#endif
Source: "{#RuntimeDir}\*"; DestDir: "{app}\runtime"; Excludes: "__pycache__,*.pyc"; Flags: recursesubdirs ignoreversion
Source: "{#ModelsDir}\*"; DestDir: "{app}\models"; Flags: recursesubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}"; Filename: "{#PythonW}"; Parameters: "-m vc.launcher"; WorkingDir: "{app}"; Comment: "Generate speech in your own voice"; IconFilename: "{app}\installer\app.ico"
Name: "{group}\User guide"; Filename: "{app}\User guide.txt"
Name: "{group}\Data folder (voices, audio, logs)"; Filename: "{%USERPROFILE}\.voice-clone"
Name: "{autodesktop}\{#AppName}"; Filename: "{#PythonW}"; Parameters: "-m vc.launcher"; WorkingDir: "{app}"; IconFilename: "{app}\installer\app.ico"; Tasks: desktopicon

[Run]
Filename: "{#PythonW}"; Parameters: "-m vc.launcher"; WorkingDir: "{app}"; Description: "Start {#AppName} now"; Flags: nowait postinstall skipifsilent runasoriginaluser

[UninstallDelete]
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\models"
Type: filesandordirs; Name: "{app}\vc"
Type: filesandordirs; Name: "{app}\web"

[Code]
function NvidiaDriverPresent: Boolean;
begin
  Result := FileExists(ExpandConstant('{sys}\nvidia-smi.exe')) or FileExists(ExpandConstant('{sys}\nvml.dll'));
end;

function InitializeSetup: Boolean;
begin
  Result := True;
  if not NvidiaDriverPresent then
    Result := SuppressibleMsgBox('No NVIDIA graphics driver was found.' + #13#10#13#10 +
      'Voice Clone Studio needs an NVIDIA GPU with 8 GB of memory (RTX 20, 30 or 40 series). ' +
      'It will not be able to generate speech on this computer.' + #13#10#13#10 +
      'Continue with the installation anyway?', mbConfirmation, MB_YESNO, idYes) = idYes;
end;
