#!/usr/bin/env python3
import argparse
import os

# Traverse the path given in the argument recursively and retain only
# a single encoding: mp3, ogg, and flac, in that order.
# Copyright: Vikram Aggarwal (c) 2021

# The filesystem is expected to contain multiple encodings in the
# *same* directory. So x.mp3, x.ogg, and x.flac are expected to be the
# same file, just with different encodings. This is how my music is
# ripped from source.


# I retain .mp3 since it is the most versatile and well-supported
# format, also out of patent protection. Then, I chose .ogg since that
# is the best quality for the smallest size. Finally, if nothing else
# is available, we choose the .flac which exists for very few albums
# that were ripped for highest quality archiving.


# TODO(viki): Add more error checking around this code. It should
# verify that we have write permissions, at least
def main(args):
    traverse(args.path, args.dryrun, args.verbose)
    print ("Main done with path %s" % args.path)


def traverse(rootDir, dryrun, verbose):
    """tranverses the directory starting at rootDir.

    Call like this:
    traverse("/home/user/music-dir", false, true)

    rootDir: String path-name, can be relative but absolute is
             probably safer.

    dryrun: if set to true, then no filesystem state is modified

    verbose: more debugging information is provided to help understand
             the functioning.

    """
    # Traverses the directory, depth-first
    for dirName, subdirList, fileList in os.walk(rootDir, topdown=False):
        # dirName is the base directory name at which all of fileList
        # are present
        if (verbose):
            print ("\n Directory: %s" % dirName)

        # Let's say the directory contains x.flac, x.ogg, y.mp3, and
        # y.ogg at the same level. Other subdirectories are irrlevant.


        # fileList continas all the files at this subdirectory level
        # only.  We need them all so we can compare against them.
        # At this point, fileList = ['x.flac', 'x.ogg', 'y.mp3', 'y.ogg']
        fileList.sort()
        if (verbose):
            print ("Evaluating list: %s" % str(fileList))

        # Names without extensions, made unique in a set. In the example above,
        # set_names = {'x', 'y'}
        set_names = { x[:x.rfind('.')] for x in fileList}
        if (verbose):
            print ("Unique names: %s" % set_names)

        # Mapping of name and extension as tuples. This contains
        # [(x,'.flac'), (x, '.ogg'), ('y', '.mp3'), ('y', '.ogg')] in
        # the example above.
        name_ext = [ (x[:x.rfind('.')], x[x.rfind('.'):]) for x in fileList ]


        # Dict of names to extensions. So 'x': ['.flac', '.ogg'] says
        # that x.flac and x.ogg are present, and 'x' is an exact match.

        # In the example above,
        # extensions = {'x': ['.flac', '.ogg'], 'y': ['.ogg', '.mp3']}
        extensions = {name: [] for name in set_names}
        for (song, encoding) in name_ext:
            extensions[song].append(encoding)

        # Now go over the extensions, deleting .flac and then .ogg
        # till the list contains a single extension.
        for song in extensions.keys():
            encodings = extensions[song]
            if (len(encodings) > 1):
                # More than one encoding, sort them in order of
                # '.mp3', '.ogg', and '.flac'. Then, delete all except
                # at position 0
                encodings.sort(key=sort_ordinal_order)

                # Now go from the back, and delete all non-zero extensions
                encodings_to_delete = encodings[1:]
                for delete_ext in encodings_to_delete:
                    file_delete = ''.join([song, delete_ext])
                    full_path = '/'.join([dirName, file_delete])

                    # Only dry-run, do not delete files
                    if (dryrun):
                        print ("Would delete: %s" % full_path)
                    else:
                        # While deleting, if verbosity is required, print what we are deleting
                        if (verbose):
                            print ("deleting: %s" % full_path)
                        os.remove(full_path)


# Sort method, which ensures that .mp3 is at first position, then
# .ogg, then .flac and then anything else.
def sort_ordinal_order(encoding):
    # mp3 is to be retained first
    if (str.lower(encoding) == '.mp3'):
        return 0
    # ogg is next
    if (str.lower(encoding) == '.ogg'):
        return 10
    # flac is last
    if (str.lower(encoding) == '.flac'):
        return 20
    # Error, how did we land here? Return a very large number and print an error
    print ("Incorrect extension (%s)" % encoding)
    return 1000


# In the commandline entrypoint, purely parse the arguments
if __name__=='__main__':
    my_parser = argparse.ArgumentParser(description='Retain a single encoding for files')

    my_parser.add_argument('-p',
                           '--path',
                           action='store',
                           type=str,
                           required=True,
                           help='the path to traverse (required)')

    my_parser.add_argument('-n',
                           '--dryrun',
                           action='store_true',
                           required=False,
                           help='If true, then don\'t perform any action, just print them')

    my_parser.add_argument('-v',
                           '--verbose',
                           action='store_true',
                           required=False,
                           help='If true, show what is going on')

    args = my_parser.parse_args()

    if (args.dryrun):
        print ("Will dry-run starting at: %s" % args.path)
    else:
        print ("Will delete files starting at: %s" % args.path)

    main(args)
