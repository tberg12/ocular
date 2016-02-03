[Taylor Berg-Kirkpatrick]: http://www.eecs.berkeley.edu/~tberg/
[Greg Durrett]: http://www.eecs.berkeley.edu/~gdurrett/
[Dan Klein]: http://www.eecs.berkeley.edu/~klein/
[Dan Garrette]: http://www.dhgarrette.com
[Hannah Alpert-Abrams]: http://www.halperta.com/



# Ocular

Ocular is a state-of-the-art historical OCR system.

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
> [Dan Garrette], [Hannah Alpert-Abrams], [Taylor Berg-Kirkpatrick], and [Dan Klein]  
> NAACL 2015


Continued development of Ocular is supported in part by a [Digital Humanities Implementation Grant](http://www.neh.gov/divisions/odh/grant-news/announcing-6-digital-humanities-implementation-grants-awards-july-2015) from the [National Endowment for the Humanities](http://www.neh.gov) for the project [Reading the First Books: Multilingual, Early-Modern OCR for Primeros Libros](https://sites.utexas.edu/firstbooks/).



## Quick-Start Guide

### Obtaining Ocular

The easiest way to get the Ocular software is to download the self-contained jar from http://www.cs.utexas.edu/~dhg/maven-repository/snapshots/edu/berkeley/cs/nlp/ocular/0.2-SNAPSHOT/ocular-0.2-SNAPSHOT-with_dependencies.jar

Once you have this jar, you will be able to run Ocular according to the instructions below in the Using Ocular section.

The jar is executable, so when you use go to use Ocular, you will run it following this template (where [MAIN-CLASS] will specify which program to run, as detailed in the Using Ocular section below):

    java -Done-jar.main.class=[MAIN-CLASS] -mx7g -jar ocular-0.2-SNAPSHOT-with_dependencies.jar [options...]

This jar includes all the necessary dependencies, so you should be able to move it to, and run it from, wherever you like.


#### Optional: Building Ocular from source code

Clone this repository, and compile the project into a jar:

    git clone https://github.com/tberg12/ocular.git
    cd ocular
    ./make_jar.sh

This creates precisely the same `ocular-0.2-SNAPSHOT-with_dependencies.jar` jar file discussed above.  Thus, this is sufficient to be able to run Ocular, as stated above, using the detailed instructions in the Using Ocular section below.

Also like above, since this jar includes all the necessary dependencies, so you should be able to move it wherever you like, without the rest of the contents of this repository.

**Compiling to an executable script instead of jar**

Alternatively, if you do not wish to create the entire jar, you can run `make_run_script.sh`, which compiles the code and generates an executable script `target/start`.  This script can be used directly, in lieu of the jar file.  Thus to run Ocular, it is sufficient to run the `make_run_script.sh` script and then use the following template instead of the template given above:
  
    export JAVA_OPTS="-mx7g"     # Increase the available memory
    target/start [MAIN-CLASS] [options...]

#### Optional: Obtaining Ocular via a dependency management system

  To incorporate Ocular into a larger project, you may use a dependency management system like Maven or SBT with the following information:

    * Repository location: http://www.cs.utexas.edu/~dhg/maven-repository/snapshots
    * Group ID: edu.berkeley.cs.nlp
    * Artifact ID: ocular
    * Version: 0.2-SNAPSHOT
    



### Using Ocular

1. Train a language model:

  Acquire some files with text written in the language(s) of your documents. For example, download a book in [English](http://www.gutenberg.org/cache/epub/2600/pg2600.txt). The path specified by `-textPath` should point to a text file or directory or directory hierarchy of text files; the path will be searched recursively for files.  Use `-lmPath` to specify where the trained LM should be written.
    
      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.TrainLanguageModel -mx7g -jar ocular-0.2-SNAPSHOT-with_dependencies.jar \
        -lmPath lm/english.lmser \
        -textPath texts/pg2600.txt

  For a multilingual (code-switching) model, specify multiple `-textPath` entries composed of a language name and a path to files containing text in that language.  For example, a combined [Spanish](https://www.gutenberg.org/cache/epub/2000/pg2000.txt)/[Latin](https://www.gutenberg.org/cache/epub/23306/pg23306.txt)/[Nahuatl](https://www.gutenberg.org/cache/epub/12219/pg12219.txt) might be trained as follows:

      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.TrainLanguageModel -mx7g -jar ocular-0.2-SNAPSHOT-with_dependencies.jar \
        -lmPath lm/trilingual.lmser \
        -textPath "spanish->texts/sp/,latin->texts/la/,nahuatl->texts/na/"

  This program will work with any languages, and any number of languages; simply add an entry for every relevant language.  The set of languages chosen should match the set of languages found in the documents that are to be transcribed.

  For older texts (either monolingual or multilingual), it might also be useful to specify the optional parameters `alternateSpellingReplacementPaths` or `-insertLongS true`, as shown here:

      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.TrainLanguageModel -mx7g -jar ocular-0.2-SNAPSHOT-with_dependencies.jar \
        -lmPath lm/trilingual.lmser \
        -textPath "spanish->texts/sp/,latin->texts/la/,nahuatl->texts/na/" \
        -alternateSpellingReplacementPaths "spanish->replace/spanish.txt,latin->replace/latin.txt,nahuatl->replace/nahuatl.txt" \
        -insertLongS true

  More details on the various command-line options can be found below.


2. Initialize a font:

  Before a font can be trained from texts, a font model consisting of a "guess" for each character must be initialized based on the fonts on your computer.  Use `-fontPath` to specify where the initialized font should be written.  Since different languages use different character sets, a language model must be given in order for the system to know what characters to initialize (`-lmPath`).

      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.InitializeFont -mx7g -jar ocular-0.2-SNAPSHOT-with_dependencies.jar \
        -lmPath lm/trilingual.lmser \
        -fontPath font/advertencias/init.fontser


3. Train a font:

  To train a font, a set of document pages must be given (`-inputPath`), along with the paths to the language model and initial font model.  Use `-outputFontPath` to specify where the trained font model should be written, and `-outputPath` to specify where transcriptions and evaluation metrics should be written.  The path specified by `-inputPath` should point to a pdf or image file or directory or directory hierarchy of such files.  The value given by `-inputPath` will be searched recursively for non-`.txt` files; the transcriptions written to the `-outputPath` will maintain the same directory hierarchy.

      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.TranscribeOrTrainFont -mx7g -jar ocular-0.2-SNAPSHOT-with_dependencies.jar \
        -trainFont true \
        -inputFontPath font/advertencias/init.fontser \
        -inputLmPath lm/trilingual.lmser \
        -inputPath sample_images/advertencias \
        -numDocs 10 \
        -outputFontPath font/advertencias/trained.fontser \
        -outputPath train_output
    
  Many more command-line options, including several that affect speed and accuracy, can be found below.


4. Transcribe some pages:

  To transcribe pages, use the same instructions as above in #3 that were used to train a font, but leave `-trainFont` unspecified (or set it to `false`).  Additionally, `-inputFontPath` should point to the newly-trained font model (instead of the "initial" font model used during font training).

      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.TranscribeOrTrainFont -mx7g -jar ocular-0.2-SNAPSHOT-with_dependencies.jar \
        -inputPath sample_images/advertencias \
        -inputFontPath font/advertencias/trained.fontser \
        -inputLmPath lm/trilingual.lmser \
        -outputPath transcribe_output 

  Many more command-line options, including several that affect speed and accuracy, can be found below.

  **Checking accuracy with a gold transcription**

  If a gold standard transcription is available for a file, it should be written in a `.txt` file in the same directory as the corresponding image, and given the same filename (but with a different extension).  These files will be used to evaluate the accuracy of the transcription (during either training or testing).  For example:

      path/to/some/image_001.jpg      # document image
      path/to/some/image_001.txt      # corresponding transcription

  For pdf files, the transcription filename is based on both the pdf filename and the relevant page number (as a 5-digit number):
  
      path/to/some/filename.pdf                 # document image
      path/to/some/filename_pdf_page00001.txt   # transcription of the document's first page






## All Command-Line Options


### TrainLanguageModel

* `-lmPath`: Output Language Model file path. 
Required.

* `-textPath`: Path to the text files (or directory hierarchies) for training the LM.  For each entry, the entire directory will be recursively searched for any files that do not start with a dot (`.`).  For a multilingual (code-switching) model, give multiple comma-separated files with language names: `"english->texts/english/,spanish->texts/spanish/,french->texts/french/"`.  Be sure to wrap the whole string with "quotes" if multiple languages are used.
Required.

* `-languagePriors`: Prior probability of each language; ignore for uniform priors. Give multiple comma-separated language/prior pairs: `"english->0.7,spanish->0.2,french->0.1"`. Be sure to wrap the whole string with "quotes".  (Only relevant if multiple languages used.) 
Default: uniform priors

* `-pKeepSameLanguage`: Prior probability of sticking with the same language when moving between words in a code-switch model transition model.  (Only relevant if multiple languages used.) 
Default: 0.999999

* `-alternateSpellingReplacementPaths`: Paths to Alternate Spelling Replacement files. If just a simple path is given, the replacements will be applied to all languages.  For language-specific replacements, give multiple comma-separated language/path pairs: `"english->rules/en.txt,spanish->rules/sp.txt,french->rules/fr.txt"`. Be sure to wrap the whole string with "quotes" if multiple languages are used. Any languages for which no replacements are needed can be safely ignored. 
Default: no replacements

* `-insertLongS`: Automatically insert \"long s\" characters into the language model training data?
Default: false

* `-removeDiacritics`: Remove diacritics? 
Default: false

* `-explicitCharacterSet`: A set of valid characters. If a character with a diacritic is found but not in this set, the diacritic will be dropped. Other excluded characters will simply be dropped. Ignore to allow all characters. *Not currently implemented*.
Default: ...

* `-charN`: LM character n-gram length.
Default: 6

* `-power`: Exponent on LM scores.
Default: 4.0

* `-lmCharCount`: Number of characters to use for training the LM.  Use 0 to indicate that the full training data should be used.
Default: 0



### InitializeFont

* `-lmPath`: Path to the language model file (so that it knows which characters to create images for).
Required.

* `-fontPath`: Output font file path.
Required.

* `-numFontInitThreads`: Number of threads to use.
Default: 8

* `-templateMaxWidthFraction`: Max template width as fraction of text line height.
Default: 1.0

* `-templateMinWidthFraction`: Min template width as fraction of text line height.
Default: 0.0

* `-spaceMaxWidthFraction`: Max space template width as fraction of text line height.
Default: 1.0

* `-spaceMinWidthFraction`: Min space template width as fraction of text line height.
Default: 0.0



### TranscribeOrTrainFont

##### Main Options

* `-inputPath`: Path of the directory that contains the input document images or pdfs. The entire directory will be recursively searched for any files that do not end in `.txt` (and that do not start with `.`).
Required.

* `-outputPath`: Path of the directory that will contain output transcriptions.
Required.

* `-inputLmPath`: Path to the input language model file.
Required.

* `-inputFontPath`: Path of the input font file.
Required.

* `-numDocs`: Number of documents (pages) to use, counting alphabetically. Ignore or use 0 to use all documents.
Default: use all documents

* `-numDocsToSkip`: Number of training documents (pages) to skip over, counting alphabetically.  Useful, in combination with -numDocs, if you want to break a directory of documents into several chunks without having to actually create separate directories.
Default: 0

* `-extractedLinesPath`: Path of the directory where the line-extraction images should be read/written.  If the line files exist here, they will be used; if not, they will be extracted and then written here.  Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document.  If ignored, the document will simply be read from the original document image file, and no line images will be written.  
Default: Don't read or write line image files.

##### Font Learning Options

* `-trainFont`: Should the font be retrained based on the input documents?
Default: false

* `-outputFontPath`: Path to write the learned font file to.
Required if trainFont is set to true, otherwise ignored.

* `-numEMIters`: Number of iterations of EM to use for font learning.
(Only relevant if trainFont is set to true.)
Default: 3

* `-updateDocBatchSize`: Number of documents to process for each parameter update.  (Only relevant if trainFont is set to true.)  This is useful if you are transcribing a large number of documents, and want to have Ocular slowly improve the model as it goes, which you would achieve with trainFont=true and numEMIter=1 (though this could also be achieved by simply running a series of smaller font training jobs each with numEMIter=1, which each subsequent job uses the model output by the previous).  Default is to update only after each full pass over the document set.

* `-accumulateBatchesWithinIter`: Should the counts from each batch accumulate with the previous batches, as opposed to each batch starting fresh?  Note that the counts will always be refreshed after a full pass through the documents.  (Only relevant if trainFont is set to true and updateDocBatchSize is used.)  Default: true
  
* `-minDocBatchSize`: The minimum number of documents that may be used to make a batch for updating parameters.  If the last batch of a pass will contain fewer than this many documents, then lump them in with the last complete batch.  Useful if you are transcribing a large number of documents, and want to have Ocular slowly improve the model as it goes.  (Only relevant if trainFont is set to true and updateDocBatchSize is used.)  Default is to always lump remaining documents in with the last complete batch.

##### Language Model Re-training Options

* `-retrainLM`: Should the language model be updated based on the input documents?  
Default: false

* `-outputLmPath`: Path to write the retrained language model file to. 
(Only relevant if retrainLM is set to true.)  
Default: Don't write out the trained LM.

##### Glyph Substitution Model Options

* `-allowGlyphSubstitution`: Should the model allow glyph substitutions? This includes substituted letters as well as letter elisions.
Default: false
	
The following options are only relevant if allowGlyphSubstitution is set to "true".

* `-inputGsmPath`: Path to the input glyph substitution model file.
Default: Don't use a pre-initialized GSM. (Learn one from scratch).

* `-gsmPower`: Exponent on GSM scores.
Default: 

* `-gsmNoCharSubPrior`: The prior probability of not-substituting the LM char. This includes substituted letters as well as letter elisions. 
Default: 0.9999999

* `-retrainGSM`: Should the glyph substitution model be updated based on the input documents?
Default: false
	
* `-outputGsmPath`: Path to write the retrained glyph substitution model file to. 
(Only relevant if retrainGSM is set to true.)
Default: Don't write out the trained GSM.
	
* `-gsmSmoothingCount`: The default number of counts that every glyph gets during training in order to smooth the glyph substitution model estimation. 
(Only relevant if retrainGSM is set to true.)
Default: 1.0
	
* `-gsmUsePrevLmChar`: Should the GSM consider (condition on) the previous LM char when deciding the glyph to output?
(Only relevant if allowGlyphSubstitution is set to true.)
Default: false

* `-gsmMinCountsForEval`: A glyph-context combination must be seen at least this many times in the last training iteration if it is to be allowed in the evaluation GSM.  This restricts spurious substitutions during evaluation.
(Only relevant if allowGlyphSubstitution is set to true.)
Default: 2

##### Line Extraction Options

* `-binarizeThreshold`: Quantile to use for pixel value thresholding. (High values mean more black pixels.)
Default: 0.12

* `-crop`: Crop pages? Useful when the image has a border around the page.
Default: true

* `-uniformLineHeight`: Scale all lines to have the same height?
Default: true

##### Miscellaneous Options

* `-emissionEngine`: Engine to use for inner loop of emission cache computation. `DEFAULT`: Uses Java on CPU, which works on any machine but is the slowest method. `OPENCL`: Faster engine that uses either the CPU or integrated GPU (depending on processor) and requires OpenCL installation. `CUDA`: Fastest method, but requires a discrete NVIDIA GPU and CUDA installation.
Default: DEFAULT

* `-beamSize`: Size of beam for Viterbi inference. (Usually in range 10-50. Increasing beam size can improve accuracy, but will reduce speed.)
Default: 10

* `-cudaDeviceID`: GPU ID when using CUDA emission engine.
Default: 0

* `-numMstepThreads`: Number of threads to use for LFBGS during m-step.
Default: 8

* `-numEmissionCacheThreads`: Number of threads to use during emission cache computation. (Only has effect when emissionEngine is set to DEFAULT.)
Default: 8

* `-numDecodeThreads`: Number of threads to use for decoding. (Should be no smaller than decodeBatchSize.)
Default: 8

* `-decodeBatchSize`: Number of lines that compose a single decode batch. (Smaller batch size can reduce memory consumption.)
Default: 32

* `-paddingMinWidth`: Min horizontal padding between characters in pixels. (Best left at default value: 1.)
Default: 1

* `-paddingMaxWidth`: Max horizontal padding between characters in pixels (Best left at default value: 5.)
Default: 5

* `-markovVerticalOffset`: Use Markov chain to generate vertical offsets. (Slower, but more accurate. Turning on Markov offsets my require larger beam size for good results.)
Default: false

* `-allowLanguageSwitchOnPunct`: A language model to be used to assign diacritics to the transcription output.
Default: true

##### Options used if evaluation should be performed during training

* `-evalInputPath`: When evaluation should be done during training, this is the path of the directory that contains the evaluation input document images. The entire directory will be recursively searched. Files that end in `.txt` will be used for as gold transcriptions.

The following options are only relevant if a value was given to -evalInputPath.

* `-evalExtractedLinesPath`: This is the path of the directory where the evaluation line-extraction images should be read/written.  If the line files exist here, they will be used; if not, they will be extracted and then written here.  Useful if: 1) you plan to run Ocular on the same documents multiple times and you want to save some time by not re-extracting the lines, or 2) you use an alternate line extractor (such as Tesseract) to pre-process the document.  If ignored, the document will simply be read from the original document image file, and no line images will be written.

* `-evalNumDocs`: When using -evalInputPath, this is the number of documents that will be evaluated on. Ignore or use 0 to use all documents. 
Default: use all documents in the specified path.

* `-evalFreq`: The font trainer will perform an evaluation every this many iterations. 
Default: Evaluate only after all iterations have completed.

* `-evalBatches`: On iterations in which we run the evaluation, should the evaluation be run after each batch (as determined by -updateBatchSize), in addition to at the end of each iteration?. 
Default: false

