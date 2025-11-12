# Installing FFmpeg on Windows

FFmpeg is required to process MP3 and M4A audio files. WAV and FLAC files work without it.

## Quick Install (Recommended)

### Option 1: Using Chocolatey
```powershell
# Install Chocolatey if you don't have it
Set-ExecutionPolicy Bypass -Scope Process -Force; [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install FFmpeg
choco install ffmpeg -y
```

### Option 2: Manual Download
1. Download FFmpeg from: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip
2. Extract the ZIP file to `C:\ffmpeg`
3. Add to PATH:
   - Open Start Menu → Search "Environment Variables"
   - Click "Environment Variables" button
   - Under "System Variables", find "Path" and click "Edit"
   - Click "New" and add: `C:\ffmpeg\bin`
   - Click OK on all windows
   - **Restart your terminal/VS Code**

### Option 3: Using WinGet (Windows 10+)
```powershell
winget install ffmpeg
```

## Verify Installation

After installation, open a **NEW** terminal and run:
```powershell
ffmpeg -version
```

You should see FFmpeg version information.

## Alternative: Use WAV Files

If you can't install FFmpeg, convert your audio files to WAV format first:
- Use online converters like https://cloudconvert.com/mp3-to-wav
- Or use Windows' built-in media converter
- Upload WAV files directly to the system

## Note

After installing FFmpeg:
1. **Restart your terminal** or VS Code
2. **Restart the server** (Ctrl+C in the server window, then run START_SERVER.bat again)
