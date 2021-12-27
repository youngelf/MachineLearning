#!/usr/bin/env python3
import os


# Traverse all the music files and print out their names.
rootDir = '/x/music'

for dirName, subdirList, fileList in os.walk(rootDir, topdown=False):
    dir_clean = dirName.encode('utf-8', 'replace').decode()
    #    print ('Directory: %s' % dir_clean) #
    music_type = None
    ogg_exists = False
    mp3_exists = False    
    flac_exists = False    
    for f_name in fileList:
        file_clean = f_name.encode('utf-8', 'replace').decode()
        if (file_clean[-4:].lower() == '.ogg'):
            music_type = 'ogg'
            # Dominant case, break early and prefer ogg
            break

        if (file_clean[-4:].lower() == '.mp3'):
            mp3_exists = True

        if (file_clean[-5:].lower() == '.flac'):
            flac_exists = True

    if (not music_type):
        if (flac_exists):
            music_type = 'flac'
        if (mp3_exists):
            music_type = 'mp3'

    if (music_type):
        print ('%s directory: %s' % (music_type, dir_clean))
        # Go through the file list, and parse just the file type here.
    else:
        print ('Not media directory: %s' % dir_clean)

    


# if .ogg exists, use those, else try to fall back to .mp3, and then to .flac.
