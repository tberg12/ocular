[Taylor Berg-Kirkpatrick]: http://www.eecs.berkeley.edu/~tberg/
[Greg Durrett]: http://www.eecs.berkeley.edu/~gdurrett/
[Dan Klein]: http://www.eecs.berkeley.edu/~klein/
[Dan Garrette]: http://www.dhgarrette.com
[Hannah Alpert-Abrams]: http://www.halperta.com/



# Ocular

Ocular is a state-of-the-art historical OCR system.

Its primary features are:

* Unsupervised learning of unknown fonts: requires only document images and a corpus of text.
* Ability to handle noisy documents: inconsistent inking, spacing, vertical alignment, etc.
* Support for multilingual documents, including those that have considerable word-level code-switching.
* Unsupervised learning of orthographic variation patterns including archaic spellings and printer shorthand.
* Simultaneous, joint transcription into both diplomatic (literal) and normalized forms.

It is described in the following publications:

> **Unsupervised Transcription of Historical Documents**
> [[pdf]](https://aclweb.org/anthology/P/P13/P13-1021.pdf)    
> [Taylor Berg-Kirkpatrick], [Greg Durrett], and [Dan Klein]  
> ACL 2013

> **Improved Typesetting Models for Historical OCR**
> [[pdf]](http://www.aclweb.org/anthology/P/P14/P14-2020.pdf)    
> [Taylor Berg-Kirkpatrick] and [Dan Klein]  
> ACL 2014

> **Unsupervised Code-Switching for Multilingual Historical Document Transcription**
> [[pdf]](http://www.aclweb.org/anthology/N15-1109)
> [[data]](https://github.com/dhgarrette/ocr-evaluation-data)  
> [Dan Garrette], [Hannah Alpert-Abrams], [Taylor Berg-Kirkpatrick], and [Dan Klein]  
> NAACL 2015

> **An Unsupervised Model of Orthographic Variation for Historical Document Transcription**
> [[pdf]](http://www.dhgarrette.com/papers/garrette_ocr_naacl2016.pdf)
> [[data]](https://github.com/dhgarrette/ocr-evaluation-data)  
> [Dan Garrette] and [Hannah Alpert-Abrams]  
> NAACL 2016


Continued development of Ocular is supported in part by a [Digital Humanities Implementation Grant](http://www.neh.gov/divisions/odh/grant-news/announcing-6-digital-humanities-implementation-grants-awards-july-2015) from the [National Endowment for the Humanities](http://www.neh.gov) for the project [Reading the First Books: Multilingual, Early-Modern OCR for Primeros Libros](https://sites.utexas.edu/firstbooks/).




## Contents of this README

1. [Quick-Start Guide](#1-quick-start-guide)
  * [Obtaining Ocular](#obtaining-ocular)
  * [Using Ocular](#using-ocular)
2. [Listing of Command-Line Options](#2-all-command-line-options)
  * [Language Model Initialization](#initializelanguagemodel)
  * [Font Initialization](#initializefont)
  * [Font Training](#trainfont)
  * [Transcription](#transcribe)


## 1. Quick-Start Guide

### Obtaining Ocular

The easiest way to get the Ocular software is to download the self-contained jar from http://www.dhgarrette.com/maven-repository/snapshots/edu/berkeley/cs/nlp/ocular/0.3-SNAPSHOT/ocular-0.3-SNAPSHOT-with_dependencies.jar

Once you have this jar, you will be able to run Ocular according to the instructions below in the [Using Ocular](#using-ocular) section; the code in this repository is not a requirement if all you'd like to do is run the software.

The jar is executable, so when you use go to use Ocular, you will run it following this template (where [MAIN-CLASS] will specify which program to run, as detailed in the [Using Ocular](#using-ocular) section below):

    java -Done-jar.main.class=[MAIN-CLASS] -mx7g -jar ocular-0.3-SNAPSHOT-with_dependencies.jar [options...]

This jar includes all the necessary dependencies, so you should be able to move it to, and run it from, wherever you like.


#### Optional: Building Ocular from source code

Clone this repository, and compile the project into a jar:

    git clone https://github.com/tberg12/ocular.git
    cd ocular
    ./make_jar.sh

This creates precisely the same `ocular-0.3-SNAPSHOT-with_dependencies.jar` jar file discussed above.  Thus, this is sufficient to be able to run Ocular, as stated above, using the detailed instructions in the [Using Ocular](#using-ocular) section below.

Also like above, since this jar includes all the necessary dependencies, so you should be able to move it wherever you like, without the rest of the contents of this repository.

**Compiling to an executable script instead of jar**

Alternatively, if you do not wish to create the entire jar, you can run `make_run_script.sh`, which compiles the code and generates an executable script `target/start`.  This script can be used directly, in lieu of the jar file.  Thus to run Ocular, it is sufficient to run the `make_run_script.sh` script and then use the following template instead of the template given above:
  
    export JAVA_OPTS="-mx7g"     # Increase the available memory
    target/start [MAIN-CLASS] [options...]

#### Optional: Obtaining Ocular via a dependency management system

  To incorporate Ocular into a larger project, you may use a dependency management system like Maven or SBT with the following information:

    Repository location: http://www.dhgarrette.com/maven-repository/snapshots
    Group ID: edu.berkeley.cs.nlp
    Artifact ID: ocular
    Version: 0.3-SNAPSHOT
    



### Using Ocular

1. Initialize a language model:

  Acquire some files with text written in the language(s) of your documents. For example, download a book in [English](http://www.gutenberg.org/cache/epub/2600/pg2600.txt). The path specified by `-inputTextPath` should point to a text file or directory or directory hierarchy of text files; the path will be searched recursively for files.  Use `-outputLmPath` to specify where the trained LM should be written.

      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.InitializeLanguageModel -mx7g -jar ocular-0.3-SNAPSHOT-with_dependencies.jar \
        -inputTextPath texts/pg2600.txt \
        -outputLmPath lm/english.lmser

  For a multilingual (code-switching) model, specify multiple `-inputTextPath` entries composed of a language name and a path to files containing text in that language.  For example, a combined [Spanish](https://www.gutenberg.org/cache/epub/2000/pg2000.txt)/[Latin](https://www.gutenberg.org/cache/epub/23306/pg23306.txt)/[Nahuatl](https://www.gutenberg.org/cache/epub/12219/pg12219.txt) might be trained as follows:

      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.InitializeLanguageModel -mx7g -jar ocular-0.3-SNAPSHOT-with_dependencies.jar \
        -inputTextPath "spanish->texts/sp/,latin->texts/la/,nahuatl->texts/na/" \
        -outputLmPath lm/trilingual.lmser

  This program will work with any languages, and any number of languages; simply add an entry for every relevant language.  The set of languages chosen should match the set of languages found in the documents that are to be transcribed.

  More details on the various command-line options can be found below.


2. Initialize a font:

  Before a font can be trained from texts, a font model consisting of a "guess" for each character must be initialized based on the fonts on your computer.  Use `-outputFontPath` to specify where the initialized font should be written.  Since different languages use different character sets, a language model must be given in order for the system to know what characters to initialize (`-inputLmPath`).

      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.InitializeFont -mx7g -jar ocular-0.3-SNAPSHOT-with_dependencies.jar \
        -inputLmPath lm/trilingual.lmser \
        -outputFontPath font/trilingual-init.fontser


3. Train a font:

  To train a font, a set of document pages must be given (`-inputDocPath`), along with the paths to the language model and initial font model.  Use `-outputFontPath` to specify where the trained font model should be written, and `-outputPath` to specify where transcriptions and (optional) evaluation metrics should be written.  The path specified by `-inputDocPath` should point to a pdf or image file or directory or directory hierarchy of such files.  The value given by `-inputDocPath` will be searched recursively for non-`.txt` files; the transcriptions written to the `-outputPath` will maintain the same directory hierarchy.

      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.TrainFont -mx7g -jar ocular-0.3-SNAPSHOT-with_dependencies.jar \
        -inputFontPath font/trilingual-init.fontser \
        -inputLmPath lm/trilingual.lmser \
        -inputDocPath sample_images/advertencias \
        -numDocs 10 \
        -outputFontPath font/advertencias/trained.fontser \
        -outputPath train_output

  Since the operation of the font trainer is to take in a font model (`-inputFontPath`) and output a new and improved font model (`-outputFontPath`), TrainFont can be run multiple times, passing the output back in as the input of the next round, to continue to making improvements.
    
  Many more command-line options, including several that affect speed and accuracy, can be found below.

  **Optional: Glyph substitution modeling for variable orthography**
  
  Ocular has the optional ability to learn, unsupervised, a mapping from archaic orthography to the orthography reflected in the trained language model. We call this a "glyph substitution model" (GSM).  To train a GSM, add the `-allowGlyphSubstitution`, `-updateGsm` and `-outputGsmPath` options.  If no `-inputGsmPath` is given, a new GSM will be created and then trained along with the font.

      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.TrainFont -mx7g -jar ocular-0.3-SNAPSHOT-with_dependencies.jar \
        -inputFontPath font/trilingual-init.fontser \
        -inputLmPath lm/trilingual.lmser \
        -inputDocPath sample_images/advertencias \
        -numDocs 10 \
        -outputFontPath font/advertencias/trained.fontser \
        -outputPath train_output \
        -allowGlyphSubstitution true \
        -updateGsm true \
        -outputGsmPath gsm/advertencias/trained.gsmser

  If `-allowGlyphSubstitution` is set to true, Ocular will produce simultaneous dual transcriptions: one *diplomatic* (literal) and one normalized to match the LM training data's orthography.


4. Transcribe some pages:

  To transcribe pages, `-inputFontPath` should point to the newly-trained font model (the `-outputFontPath` from the training step, instead of the "initial" font model used during font training).

      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.Transcribe -mx7g -jar ocular-0.3-SNAPSHOT-with_dependencies.jar \
        -inputDocPath sample_images/advertencias \
        -inputLmPath lm/trilingual.lmser \
        -inputFontPath font/advertencias/trained.fontser \
        -outputPath transcribe_output 

  As above, if `-allowGlyphSubstitution` is set to true and the `-inputGsmPath` is given, Ocular will produce simultaneous dual transcriptions: one *diplomatic* (literal) and one normalized to match the LM training data's orthography.

  Many more command-line options, including several that affect speed and accuracy, can be found below.  Among these, `-skipAlreadyTranscribedDocs` might be particularly useful.
  
  **Optional: Continued model improvements during transcription**
  
  Since training is a model is done in an unsupervised fashion (it requires no gold transcriptions), the operation of transcribing is actually a subset of EM font training.  Because of this, it is possible make further improvements to the models during transcription, without having to make multiple iterations over the documents.  This can be done by setting `-updateFont` to `true`, and `-updateDocBatchSize` to a reasonable number of training documents:

      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.Transcribe -mx7g -jar ocular-0.3-SNAPSHOT-with_dependencies.jar \
        -inputDocPath sample_images/advertencias \
        -inputLmPath lm/trilingual.lmser \
        -inputFontPath font/advertencias/trained.fontser \
        -outputPath transcribe_output \
        -updateFont true \
        -updateDocBatchSize 50 \
        -outputFontPath font/advertencias/trained.fontser
      
  The same can be done to update the glyph substitution model by passing in the previously-trained model (`-inputGsmPath`) and setting `-updateGsm` to `true`.

        -allowGlyphSubstitution true \
        -inputGsmPath gsm/advertencias/trained.gsmser \
        -updateGsm true \
        -outputGsmPath gsm/advertencias/trained.gsmser
  
  **Optional: Checking accuracy with a gold transcription**

  If a gold standard transcription is available for a file, it should be written in a `.txt` file in the same directory as the corresponding image, and given the same filename (but with a different extension).  These files will be used to evaluate the accuracy of the transcription (during either training or testing).  Likewise, if a gold normalized transcription is available, it should be given the same filename, but with `_normalized` appended.  For example:

      path/to/some/image_001.jpg              # document image
      path/to/some/image_001.txt              # corresponding transcription
      path/to/some/image_001_normalized.txt   # corresponding normalized transcription

  For pdf files, the transcription filename is based on both the pdf filename and the relevant page number (as a 5-digit number):

      path/to/some/filename.pdf                            # document image
      path/to/some/filename_pdf_page00001.txt              # transcription of the document's first page
      path/to/some/filename_pdf_page00001_normalized.txt   # corresponding normalized transcription






## 2. All Command-Line Options

### InitializeLanguageModel

##### Required

* `-inputTextPath`:
Path to the text files (or directory hierarchies) for training the LM.  For each entry, the entire directory will be recursively searched for any files that do not start with `.`.  For a multilingual (code-switching) model, give multiple comma-separated files with language names: "english->texts/english/,spanish->texts/spanish/,french->texts/french/".  Be sure to wrap the whole string with "quotes".)
Required.

* `-outputLmPath`:
Output LM file path.
Required.

##### Additional Options

* `-minCharCount`:
Number of times the character must be seen in order to be included.
Default: 10

* `-insertLongS`:
Automatically insert "long s" characters into the language model training data?
Default: false

* `-charNgramLength`:
LM character n-gram length. If just one language is used, or if all languages should use the same value, just give an integer.  If languages can have different values, give them as comma-separated language/integer pairs: "english->6,spanish->4,french->4"; be sure to wrap the whole string with "quotes".
Default: 6

* `-alternateSpellingReplacementPaths`:
Paths to Alternate Spelling Replacement files. If just a simple path is given, the replacements will be applied to all languages.  For language-specific replacements, give multiple comma-separated language/path pairs: "english->rules/en.txt,spanish->rules/sp.txt,french->rules/fr.txt". Be sure to wrap the whole string with "quotes". Any languages for which no replacements are need can be safely ignored.
Default: No alternate spelling replacements.

##### Rarely Used Options

* `-removeDiacritics`:
Remove diacritics?
Default: false

* `-pKeepSameLanguage`:
Prior probability of sticking with the same language when moving between words in a code-switch model transition model. (Only relevant if multiple languages used.)
Default: 0.999999

* `-languagePriors`:
Prior probability of each language; ignore for uniform priors. Give multiple comma-separated language/prior pairs: "english->0.7,spanish->0.2,french->0.1". Be sure to wrap the whole string with "quotes". (Only relevant if multiple languages used.)
Default: Uniform priors.

* `-lmPower`:
Exponent on LM scores.
Default: 4.0

* `-explicitCharacterSet`:
A set of valid characters. If a character with a diacritic is found but not in this set, the diacritic will be dropped. Other excluded characters will simply be dropped. Ignore to allow all characters.
Default: Allow all characters.

* `-lmCharCount`:
Number of characters to use for training the LM.  Use 0 to indicate that the full training data should be used.
Default: Use all documents in full.

### InitializeFont

##### Required

* `-inputLmPath`:
Path to the language model file (so that it knows which characters to create images for).
Required.

* `-outputFontPath`:
Output font file path.
Required.

##### Additional Options

* `-allowedFontsPath`:
Path to a file that contains a custom list of font names that may be used to initialize the font. The file should contain one font name per line.
Default: Use all valid fonts found on the computer.

##### Rarely Used Options

* `-numFontInitThreads`:
Number of threads to use.
Default: 8

* `-spaceMaxWidthFraction`:
Max space template width as fraction of text line height.
Default: 1.0

* `-spaceMinWidthFraction`:
Min space template width as fraction of text line height.
Default: 0.0

* `-templateMaxWidthFraction`:
Max template width as fraction of text line height.
Default: 1.0

* `-templateMinWidthFraction`:
Min template width as fraction of text line height.
Default: 0.0

### TrainFont

##### Main Options

* `-inputDocPath`:
Path to the directory that contains the input document images. The entire directory will be searched recursively for any files that do not end in `.txt` (and that do not start with `.`).  Files will be processed in lexicographical order.
Default: Either inputDocPath or inputDocListPath is required.

* `-inputDocListPath`:
Path to a file that contains a list of paths to images files that should be used.  The file should contain one path per line. These paths will be searched in order.  Each path may point to either a file or a directory, which will be searched recursively for any files that do not end in `.txt` (and that do not start with `.`).  Paths will be processed in the order given in the file, and each path will be searched in lexicographical order.
Default: Either inputDocPath or inputDocListPath is required.

* `-inputFontPath`:
Path of the input font file.
Required.

* `-inputLmPath`:
Path to the input language model file.
Required.

* `-numDocs`:
Number of documents (pages) to use, counting alphabetically. Ignore or use 0 to use all documents.
Default: Use all documents.

* `-numDocsToSkip`:
Number of training documents (pages) to skip over, counting alphabetically.  Useful, in combination with -numDocs, if you want to break a directory of documents into several chunks.
Default: 0

* `-numEMIters`:
Number of iterations of EM to use for font learning.
Default: 3

* `-continueFromLastCompleteIteration`:
If true, the font trainer will find the latest completed iteration in the outputPath and load it in order to pick up training from that point.  Convenient if a training run crashes when only partially completed.
Default: false

* `-outputPath`:
Path of the directory that will contain output transcriptions.
Required.

* `-outputFormats`:
Output formats to be generated. Choose from one or multiple of {dipl,norm,normlines,comp,html,alto}, comma-separated.  dipl = diplomatic, norm = normalized (lines joined), normlines = normalized (separate lines), comp = comparisons.
Default: dipl,norm if -allowGlyphSubstitution=true; dipl otherwise.

* `-outputFontPath`:
Path to write the learned font file to.
Required if updateFont is set to true, otherwise ignored.

##### Additional Options

* `-extractedLinesPath`:
Path of the directory where the line-extraction images should be read/written.  If the line files exist here, they will be used; if not, they will be extracted and then written here.  Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document.  If ignored, the document will simply be read from the original document image file, and no line images will be written.
Default: Don't read or write line image files.

* `-updateDocBatchSize`:
Number of documents to process for each parameter update.  This is useful if you are transcribing a large number of documents, and want to have Ocular slowly improve the model as it goes, which you would achieve with updateFont=true.
Default: Update only after each full pass over the document set.

These options affect the speed of font training

* `-emissionEngine`:
Engine to use for inner loop of emission cache computation. `DEFAULT`: Uses Java on CPU, which works on any machine but is the slowest method. `OPENCL`: Faster engine that uses either the CPU or integrated GPU (depending on processor) and requires OpenCL installation. `CUDA`: Fastest method, but requires a discrete NVIDIA GPU and CUDA installation.
Default: DEFAULT

* `-beamSize`:
Size of beam for Viterbi inference. (Usually in range 10-50. Increasing beam size can improve accuracy, but will reduce speed.)
Default: 10

* `-markovVerticalOffset`:
Use Markov chain to generate vertical offsets. (Slower, but more accurate. Turning on Markov offsets my require larger beam size for good results.)
Default: false

##### Glyph Substitution Model Options

Glyph substitution is the feature that allows Ocular to use a probabilistic mapping from modern orthography (as used in the language model training text) to the orthography seen in the documents. If the glyph substitution feature is used, Ocular will jointly produce dual transcriptions: one that is an exact transcription of the document, and one that is a normalized version of the text.

* `-allowGlyphSubstitution`:
Should the model allow glyph substitutions? This includes substituted letters as well as letter elisions.
Default: false

* `-inputGsmPath`:
Path to the input glyph substitution model file. (Only relevant if allowGlyphSubstitution is set to true.)
Default: Don't use a pre-initialized GSM. (Learn one from scratch).

* `-updateGsm`:
Should the glyph substitution model be trained (or updated) along with the font? (Only relevant if allowGlyphSubstitution is set to true.)
Default: false

* `-outputGsmPath`:
Path to write the retrained glyph substitution model file to.
Required if updateGsm is set to true, otherwise ignored.

##### Language Model Training Options

* `-updateLM`:
Should the language model be updated along with the font?
Default: false

* `-outputLmPath`:
Path to write the retrained language model file to.
Required if updateLM is set to true, otherwise ignored.

##### Line Extraction Options

* `-binarizeThreshold`:
Quantile to use for pixel value thresholding. (High values mean more black pixels.)
Default: 0.12

* `-crop`:
Crop pages?
Default: true

##### Evaluate During Training

* `-evalInputDocPath`:
When evaluation should be done during training (after each parameter update in EM), this is the path of the directory that contains the evaluation input document images. The entire directory will be recursively searched for any files that do not end in `.txt` (and that do not start with `.`). (Only relevant if updateFont is set to true.)
Default: Do not evaluate during font training.

* `-evalNumDocs`:
When using -evalInputDocPath, this is the number of documents that will be evaluated on. Ignore or use 0 to use all documents.
Default: Use all documents in the specified path.

* `-evalExtractedLinesPath`:
When using -evalInputDocPath, this is the path of the directory where the evaluation line-extraction images should be read/written.  If the line files exist here, they will be used; if not, they will be extracted and then written here.  Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document.  If ignored, the document will simply be read from the original document image file, and no line images will be written.
Default: Don't read or write line image files.

* `-evalFreq`:
When using -evalInputDocPath, the font trainer will perform an evaluation every `evalFreq` iterations.
Default: Evaluate only after all iterations have completed.

* `-evalBatches`:
When using -evalInputDocPath, on iterations in which we run the evaluation, should the evaluation be run after each batch, as determined by -updateDocBatchSize (in addition to after each iteration)?
Default: false

##### Rarely Used Options

* `-allowLanguageSwitchOnPunct`:
A language model to be used to assign diacritics to the transcription output.
Default: true

* `-cudaDeviceID`:
GPU ID when using CUDA emission engine.
Default: 0

* `-decodeBatchSize`:
Number of lines that compose a single decode batch. (Smaller batch size can reduce memory consumption.)
Default: 32

* `-gsmElideAnything`:
Should the GSM be allowed to elide letters even without the presence of an elision-marking tilde?
Default: false

* `-gsmElisionSmoothingCountMultiplier`:
gsmElisionSmoothingCountMultiplier.
Default: 100.0

* `-gsmNoCharSubPrior`:
The prior probability of not-substituting the LM char. This includes substituted letters as well as letter elisions.
Default: 0.9

* `-gsmPower`:
Exponent on GSM scores.
Default: 4.0

* `-gsmSmoothingCount`:
The default number of counts that every glyph gets in order to smooth the glyph substitution model estimation.
Default: 1.0

* `-paddingMaxWidth`:
Max horizontal padding between characters in pixels (Best left at default value.)
Default: 5

* `-paddingMinWidth`:
Min horizontal padding between characters in pixels. (Best left at default value.)
Default: 1

* `-uniformLineHeight`:
Scale all lines to have the same height?
Default: true

* `-numDecodeThreads`:
Number of threads to use for decoding. (More thread may increase speed, but may cause a loss of continuity across lines.)
Default: 1

* `-numEmissionCacheThreads`:
Number of threads to use during emission cache computation. (Only has effect when emissionEngine is set to DEFAULT.)
Default: 8

* `-numMstepThreads`:
Number of threads to use for LFBGS during m-step.
Default: 8

### Transcribe

##### Main Options

* `-inputDocPath`:
Path to the directory that contains the input document images. The entire directory will be searched recursively for any files that do not end in `.txt` (and that do not start with `.`).  Files will be processed in lexicographical order.
Default: Either inputDocPath or inputDocListPath is required.

* `-inputDocListPath`:
Path to a file that contains a list of paths to images files that should be used.  The file should contain one path per line. These paths will be searched in order.  Each path may point to either a file or a directory, which will be searched recursively for any files that do not end in `.txt` (and that do not start with `.`).  Paths will be processed in the order given in the file, and each path will be searched in lexicographical order.
Default: Either inputDocPath or inputDocListPath is required.

* `-inputFontPath`:
Path of the input font file.
Required.

* `-inputLmPath`:
Path to the input language model file.
Required.

* `-numDocs`:
Number of documents (pages) to use, counting alphabetically. Ignore or use 0 to use all documents.
Default: Use all documents.

* `-numDocsToSkip`:
Number of training documents (pages) to skip over, counting alphabetically.  Useful, in combination with -numDocs, if you want to break a directory of documents into several chunks.
Default: 0

* `-skipAlreadyTranscribedDocs`:
If true, for each doc the outputPath will be checked for an existing transcription and if one is found then the document will be skipped.
Default: false

* `-outputPath`:
Path of the directory that will contain output transcriptions.
Required.

* `-outputFormats`:
Output formats to be generated. Choose from one or multiple of {dipl,norm,normlines,comp,html,alto}, comma-separated.  dipl = diplomatic, norm = normalized (lines joined), normlines = normalized (separate lines), comp = comparisons.
Default: dipl,norm if -allowGlyphSubstitution=true; dipl otherwise.

##### Additional Options

* `-extractedLinesPath`:
Path of the directory where the line-extraction images should be read/written.  If the line files exist here, they will be used; if not, they will be extracted and then written here.  Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document.  If ignored, the document will simply be read from the original document image file, and no line images will be written.
Default: Don't read or write line image files.

* `-failIfAllDocsAlreadyTranscribed`:
If true, an exception will be thrown if all of the input documents have already been transcribed (and thus the job has nothing to do).  Ignored unless -skipAlreadyTranscribedDocs=true.
Default: false

These options affect the speed of transcription

* `-emissionEngine`:
Engine to use for inner loop of emission cache computation. `DEFAULT`: Uses Java on CPU, which works on any machine but is the slowest method. `OPENCL`: Faster engine that uses either the CPU or integrated GPU (depending on processor) and requires OpenCL installation. `CUDA`: Fastest method, but requires a discrete NVIDIA GPU and CUDA installation.
Default: DEFAULT

* `-beamSize`:
Size of beam for Viterbi inference. (Usually in range 10-50. Increasing beam size can improve accuracy, but will reduce speed.)
Default: 10

* `-markovVerticalOffset`:
Use Markov chain to generate vertical offsets. (Slower, but more accurate. Turning on Markov offsets my require larger beam size for good results.)
Default: false

##### Glyph Substitution Model Options

Glyph substitution is the feature that allows Ocular to use a probabilistic mapping from modern orthography (as used in the language model training text) to the orthography seen in the documents. If the glyph substitution feature is used, Ocular will jointly produce dual transcriptions: one that is an exact transcription of the document, and one that is a normalized version of the text.

* `-allowGlyphSubstitution`:
Should the model allow glyph substitutions? This includes substituted letters as well as letter elisions.
Default: false

* `-inputGsmPath`:
Path to the input glyph substitution model file. (Only relevant if allowGlyphSubstitution is set to true.)
Default: Don't use a pre-initialized GSM. (Learn one from scratch).

##### Model Updating Options

* `-updateDocBatchSize`:
Number of documents to process for each parameter update.  This is useful if you are transcribing a large number of documents, and want to have Ocular slowly improve the model as it goes, which you would achieve with updateFont=true.
Default: Update only after each full pass over the document set.

For updating the font model

* `-updateFont`:
Update the font during transcription based on the new input documents?
Default: false

* `-outputFontPath`:
Path to write the learned font file to.
Required if updateFont is set to true, otherwise ignored.

For updating the glyph substitution model

* `-updateGsm`:
Should the glyph substitution model be trained (or updated) along with the font? (Only relevant if allowGlyphSubstitution is set to true.)
Default: false

* `-outputGsmPath`:
Path to write the retrained glyph substitution model file to.
Required if updateGsm is set to true, otherwise ignored.

For updating the language model

* `-updateLM`:
Should the language model be updated along with the font?
Default: false

* `-outputLmPath`:
Path to write the retrained language model file to.
Required if updateLM is set to true, otherwise ignored.

##### Line Extraction Options

* `-binarizeThreshold`:
Quantile to use for pixel value thresholding. (High values mean more black pixels.)
Default: 0.12

* `-crop`:
Crop pages?
Default: true

##### Evaluate During Training

* `-evalInputDocPath`:
When evaluation should be done during training (after each parameter update in EM), this is the path of the directory that contains the evaluation input document images. The entire directory will be recursively searched for any files that do not end in `.txt` (and that do not start with `.`). (Only relevant if updateFont is set to true.)
Default: Do not evaluate during font training.

* `-evalNumDocs`:
When using -evalInputDocPath, this is the number of documents that will be evaluated on. Ignore or use 0 to use all documents.
Default: Use all documents in the specified path.

* `-evalBatches`:
When using -evalInputDocPath, on iterations in which we run the evaluation, should the evaluation be run after each batch, as determined by -updateDocBatchSize (in addition to after each iteration)?
Default: false

* `-evalExtractedLinesPath`:
When using -evalInputDocPath, this is the path of the directory where the evaluation line-extraction images should be read/written.  If the line files exist here, they will be used; if not, they will be extracted and then written here.  Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document.  If ignored, the document will simply be read from the original document image file, and no line images will be written.
Default: Don't read or write line image files.

##### Rarely Used Options

* `-allowLanguageSwitchOnPunct`:
A language model to be used to assign diacritics to the transcription output.
Default: true

* `-cudaDeviceID`:
GPU ID when using CUDA emission engine.
Default: 0

* `-decodeBatchSize`:
Number of lines that compose a single decode batch. (Smaller batch size can reduce memory consumption.)
Default: 32

* `-gsmElideAnything`:
Should the GSM be allowed to elide letters even without the presence of an elision-marking tilde?
Default: false

* `-gsmElisionSmoothingCountMultiplier`:
gsmElisionSmoothingCountMultiplier.
Default: 100.0

* `-gsmNoCharSubPrior`:
The prior probability of not-substituting the LM char. This includes substituted letters as well as letter elisions.
Default: 0.9

* `-gsmPower`:
Exponent on GSM scores.
Default: 4.0

* `-gsmSmoothingCount`:
The default number of counts that every glyph gets in order to smooth the glyph substitution model estimation.
Default: 1.0

* `-paddingMaxWidth`:
Max horizontal padding between characters in pixels (Best left at default value.)
Default: 5

* `-paddingMinWidth`:
Min horizontal padding between characters in pixels. (Best left at default value.)
Default: 1

* `-uniformLineHeight`:
Scale all lines to have the same height?
Default: true

* `-numDecodeThreads`:
Number of threads to use for decoding. (More thread may increase speed, but may cause a loss of continuity across lines.)
Default: 1

* `-numEmissionCacheThreads`:
Number of threads to use during emission cache computation. (Only has effect when emissionEngine is set to DEFAULT.)
Default: 8

* `-numMstepThreads`:
Number of threads to use for LFBGS during m-step.
Default: 8
