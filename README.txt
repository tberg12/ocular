Ocular can recognize collections of documents that use historical fonts.
The system works best when you run it on a collection of documents that
mainly uses a single font (e.g. a whole book). Ocular works in two phases:

///////////////////////////////////////////////////////////////////////////

Phase 1: Train a font model for your particular collection of documents

Ocular is unsupervised, so you don't need document images that are
labeled with human transcriptions in order to learn the font. Instead,
Ocular learns the font directly, straight from the set of input document
images. Since the system often only needs a small portion of a book (on
the order of 10 pages) to effectively learn the font, it is usually most
efficient to learn the font from a subset of your document collection. So,
the first step is to select about 10 document images from your collection
that are relatively clean (no weird formatting, and minimal scanning
noise). Put these document images togther in a directory. Run the
following command from within the ocular directory in order to learn a
font model:

java -mx7g -jar ocular.jar ++conf/base.conf -learnFont true -inputPath selected_images_path -outputPath output_path -outputFontPath output_font_path

Here, 'selected_images_path' is the path to the directory where you put
your selected document images, 'output_path' is a directory where the
output transcriptions (useful for checking whether you've effectively
learned the font) and images of the line extractions (useful to see if
the pre-processing phase was effective) will be written, and
'output_font_path' is the path of the file that will contain the learned
font model.

The directory './test' contains several images of historical documents
that can be used to test Ocular. Simply run the command above with
'-inputPath' set to './test'. 

///////////////////////////////////////////////////////////////////////////

Phase 2: Transcribe your full document collection using the learned font

Now that you've learned a font model (the one that was written to
'output_font_path' in phase 1) you can transcribe your full collection of 
document images. In order to perform the full transcription, run the
following command from within the ocular directory:

java -mx7g -jar ocular.jar ++conf/base.conf -learnFont false -initFontPath output_font_path -inputPath all_images_path -outputPath output_path

Here, 'output_font_path' is the path to the font model you produced in 
phase 1, 'all_images_path' is a path to the directory containing all the
images in your document collection, and 'output_path' specifies the path 
to a directory where you want to store the output transcriptions and the
line extraction images for each document.

///////////////////////////////////////////////////////////////////////////

Configuration parameters:

In order to print the full list of options for running Ocular use the
following command:

java -jar ocular.jar -help

///////////////////////////////////////////////////////////////////////////

Important configuration parameters that affect accuracy:

An important option that can be specified is the choice of language model.
The 'lm' directory contains several language model files.

'./lm/nyt.lmser'
-Trained on New York Times data.

'./lm/nyt_longs.lmser'
-Trained on New York Times data.
-Uses long s character, suitable for documents that use the long s glyph.

'./lm/ob.lmser'
-Trained on Old Bailey historical court proceedings.

'./lm/ob_longs.lmser'
-Trained on Old Bailey historical court proceedings.
-Uses long s character, suitable for documents that use the long s glyph.

The language model used by Ocular is specified by the 'lmPath' parameter,
which can either be set from the command line with '-lmPath' or modified
in the configuration file './conf/base.conf'. The default setting of
this parameter is './lm/nyt.lmser'.

Another important option is the threshold for binarizing the input images
into black and white pixels. This option is specified with the
'binarizeThreshold' parameter, again either on the command line, or directly
in the configuration file. It can be set to real numbers between 0 and 1,
and has a default value of 0.12. If you notice that the line extraction
images in the output folder look over-exposed (too much white), try raising
the binarization threshold. This can improve accuracy.

///////////////////////////////////////////////////////////////////////////

Important configuration parameters that affect speed:

Ocular can be computationally intensive to run. It requires a machine
with at least 8GB of RAM, and can benefit from more advanced hardware like
a discrete GPU. 

On a high-end modern desktop computer, Ocular should take approximately
1 minute per page during the transcription phase (phase 2). On an older
laptop, Ocular may take as much as 10 minutes per page during the same
phase.

It is possible to run Ocular in a fast mode that sacrifies a small amount
of transcription accuracy in order to increase speed. In order to run in
this mode replace every mention of the '++conf/base.conf' configuration
file with '++conf/fast.conf' in the description of phase 1 and phase 2
above.

It is also possible to substantially speed up Ocular without sacrificing
accuracy by making better use of available hardware. If you are running
on an Apple laptop try setting the 'emissionEngine' parameter to
'OPENCL'. This can be done either with the command line flag
'-emissionEngine' or from within the configuration file. This option will
make use of the integrated GPU where possible, and make more efficient
use of the processor otherwise. Specifically, using the following commands
for phase 1 and phase 2:

Phase 1:

java -mx7g -jar ocular.jar ++conf/base.conf -emissionEngine OPENCL -learnFont true -inputPath selected_images_path -outputPath output_path -outputFontPath output_font_path

Phase 2:

java -mx7g -jar ocular.jar ++conf/base.conf -emissionEngine OPENCL -learnFont false -initFontPath output_font_path -inputPath all_images_path -outputPath output_path

If you are not using an Apple machine, the 'OPENCL' option may still be
used if you first install OpenCL. Instructions for how to do this can be
found online.

Finally, if you have a discrete NVIDIA GPU, Ocular can be sped up even
more. You need to make sure CUDA is installed, and then set
'emissionEngine' to 'CUDA'. You may need to add the corresponding JCuda
DLL to your dynamic library path when you run the java command. The DLLs
are included in the './lib' directory. Choose the one that corresponds
to your operating system.

///////////////////////////////////////////////////////////////////////////

Text line extraction:

OCR usually has two steps: 1) extract images of isolated lines of text from
the input image, 2) recognize each line image. Strictly speaking, Ocular
is a text line recognizer.

The line extractor Ocular uses is kind of rudimentary and can only handle
documents without complex formatting (i.e. only a single column of text).
It should, however, be pretty robust to noise and may even perform well on
book images that contain some of the opposing page in the margin. The way
to check whether something did go wrong in the line extraction phase is to
look at the line extraction images in the output directory. If these don't
contain a sequence of images, each of which contains a single line of text,
something went wrong.

If something does go wrong here, there are several possible solutions. One
solution is to manually crop your images ahead of time (not into individual
lines, just into big blocks of text). But that can be time consuming. A
better option is something we're working on right now and will release in
an update to Ocular. The open source OCR library called Ocropus contains a
high quality line extractor. In a future release you will have the option
to use the Ocropus line extractor from within our historical recognizer.
This way you'd get the best of both worlds: a great historical text
recognizer and a line extractor that can handle more complicated
formatting.

///////////////////////////////////////////////////////////////////////////

License:

Copyright (c) 2013 Taylor Berg-Kirkpatrick and Greg Durrett. All Rights
Reserved.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

///////////////////////////////////////////////////////////////////////////

Update Log:

2014-8-23
--Preliminary version.


