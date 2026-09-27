VOICE CLONE STUDIO - USER GUIDE
================================

Generate speech in your own voice from any script, on your own computer.
Nothing you record is uploaded anywhere.


WHAT YOU NEED
-------------
  - Windows 10 or 11
  - An NVIDIA graphics card with 8 GB of memory (RTX 20, 30, 40 or 50 series)
    with NVIDIA driver version 572 or newer
  - About 15 GB of disk space (30 GB free while installing)
  - A microphone, or a recording of your voice


INSTALLING
----------
  1. Put the setup .exe and ALL the .bin files in the same folder.
  2. Run the .exe. If Windows shows "Windows protected your PC",
     click "More info" and then "Run anyway".
  3. Start Voice Clone Studio from the Start menu or the desktop.

THE FIRST RUN IS SLOW - THIS IS NORMAL
  Right after installing, Windows checks every new file the first time it is
  used. The first recording analysis can take 5 to 8 minutes and the first
  generation 3 to 5 minutes. Let it finish. After that, an analysis takes
  about 2 minutes and a few sentences generate in 10 to 30 seconds.


STEP 1 - CREATE YOUR VOICE
--------------------------
Click "Upload Audio" (WAV, MP3, FLAC, M4A) or "Record sample".

For the best result, the recording should be:
  - 5 to 10 minutes long
  - only you speaking, no other voices, no music
  - in a quiet room, with the microphone 15 to 20 cm from your mouth
  - natural speech: read an article or tell a story, with long and short
    sentences, in the language you want to generate
  - made with the same microphone from start to finish

You get a quality score and, if something is wrong, advice on how to fix it.
A score below 55%, more than one speaker, or less than a minute of speech is
rejected. Light background noise is removed automatically.

Type a name and click "Create Voice Profile".


CONFIRM IT IS YOUR VOICE
------------------------
Read the statement on screen aloud, including the three code words, and click
"Stop" when you are done. The app checks that the words match, that the voice
matches your recording, and that the recording is live.

If it fails, click "New statement" and read again, slowly and clearly.
If the microphone does not work, allow microphone access when the window asks.


STEP 2 - ENTER YOUR SCRIPT
--------------------------
Type or paste the text. You can write:
  - Hindi in Devanagari:        आज हम एक नई शुरुआत करेंगे।
  - Hinglish in Roman letters:  Aaj hum ek nayi shuruaat karenge.
  - English

Under the box you see exactly what will be spoken. If a word was converted
wrongly, click "Edit this text" and correct it. Hindi written in Devanagari
gives the most reliable pronunciation.

Style changes the delivery, Speed makes it faster or slower.
Click "Generate Voice". A few sentences take 10 to 30 seconds.


STEP 3 - PREVIEW AND DOWNLOAD
-----------------------------
Listen, then "Download WAV" (best quality) or "Download MP3" (smaller).
"Regenerate" makes a new take of the same text; each take is slightly different.

"Speaker similarity" compares the result with your own recordings. If it is
marked low, try Regenerate, another style, or a script in the language of your
recording.


MAKING IT SOUND MORE LIKE YOU (optional)
----------------------------------------
Open the "Voices" tab and click "Fine-tune this voice". It takes a few minutes.
The result is only kept if it really sounds more like you; otherwise it is
marked "rejected" and the normal voice is used.


CHECKING A FILE
---------------
The "Verify audio" tab tells you whether a file was made by this app or
carries an AI-speech watermark.


DELETING YOUR VOICE
-------------------
"Voices" tab, "Delete voice". The voice data is erased from your computer.
Your voices, generated audio and logs are stored in the folder
  C:\Users\<you>\.voice-clone
Uninstalling the app does not delete that folder.


RESPONSIBLE USE
---------------
Only create a voice from your own recording, or with the clear agreement of
the speaker. Never use generated speech to impersonate, deceive, harass or
defraud anyone. Every file is watermarked and labelled as AI-generated.


IF SOMETHING GOES WRONG
-----------------------
"CUDA out of memory" or generation fails
    Close games and other programs that use the graphics card, then try again.

The window does not open
    Wait a minute and start the app again. If it still fails, send the files in
    C:\Users\<you>\.voice-clone\logs

The status at the top says "Worker starting..." for more than two minutes
    Close the window and start the app again.

The microphone buttons do nothing
    Allow microphone access for the app window, or record with another program
    and use "Upload Audio".
