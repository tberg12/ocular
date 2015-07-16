[Taylor Berg-Kirkpatrick]: http://www.eecs.berkeley.edu/~tberg/
[Greg Durrett]: http://www.eecs.berkeley.edu/~gdurrett/
[Dan Klein]: http://www.eecs.berkeley.edu/~klein/
[Dan Garrette]: http://www.dhgarrette.com
[Hannah Alpert-Abrams]: http://www.halperta.com/



# Ocular

Ocular is a state-of-the-art historical OCR system.

It is described in the following publications:

> Unsupervised Transcription of Historical Documents [[pdf]](https://aclweb.org/anthology/P/P13/P13-1021.pdf)    
> [Taylor Berg-Kirkpatrick], [Greg Durrett], and [Dan Klein]  
> ACL 2013

> Improved Typesetting Models for Historical OCR [[pdf]](http://www.aclweb.org/anthology/P/P14/P14-2020.pdf)    
> [Taylor Berg-Kirkpatrick] and [Dan Klein]  
> ACL 2014

> Unsupervised Code-Switching for Multilingual Historical Document Transcription [[pdf]](http://www.aclweb.org/anthology/N15-1109)    
> [Dan Garrette], [Hannah Alpert-Abrams], [Taylor Berg-Kirkpatrick], and [Dan Klein]  
> NAACL 2015



## Quick-Start Guide

### Getting set up

Clone this repository, and compile the project into a jar:

    git clone https://github.com/tberg12/ocular.git
    cd ocular
    ./compile.sh

  This creates a jar file `ocular-0.2-SNAPSHOT-with_dependencies.jar` that can be run like:
  
    java -Done-jar.main.class=[MAIN-CLASS] -mx7g -jar ocular-0.2-SNAPSHOT-with_dependencies.jar [options...]

  This jar includes all the necessary dependencies, so you should be able to move it wherever you like, without the rest of the contents of this repository.

  Alternatively, if you do not require the source code or sample files, the jar file can just be downloaded from here: [http://www.cs.utexas.edu/~dhg/maven-repository/snapshots/edu/berkeley/cs/nlp/ocular/0.2-SNAPSHOT/ocular-0.2-SNAPSHOT.jar].

#### Other ways of obtaining Ocular

  The `compile.sh` script also generates an executable script `target/start` that can be run like:
  
    export JAVA_OPTS="-mx7g"     # Increase the available memory
    target/start [MAIN-CLASS] [options...]

  Alternatively, to incorporate Ocular into a larger project, you may use a dependency management system like Maven or SBT with the following information:

    * Repository location: `http://www.cs.utexas.edu/~dhg/maven-repository/snapshots`
    * Group ID: `edu.berkeley.cs.nlp`
    * Artifact ID: `ocular`
    * Version: `0.2-SNAPSHOT`
    



### Using Ocular

1. Train a language model:

  Acquire some files with text written in the language(s) of your documents. For example, download a book in [English](http://www.gutenberg.org/cache/epub/2600/pg2600.txt). The path specified by `-textPath` should point to a text file or directory or directory hierarchy of text files; the path will be searched recursively for files.  Use `-lmPath` to specify where the trained LM should be written.
    
      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.TrainLanguageModel -mx7g -jar ocular-0.2-SNAPSHOT-with_dependencies.jar \
        -lmPath lm/english.lmser \
        -textPath texts/pg2600.txt

  For a multilingual (code-switching) model, specify multiple `-textPath` entries composed of a language name and a path to files containing text in that language.  For example, a combined [Spanish](https://www.gutenberg.org/cache/epub/2000/pg2000.txt)/[Latin](https://www.gutenberg.org/cache/epub/23306/pg23306.txt)/[Nahuatl](https://www.gutenberg.org/cache/epub/12219/pg12219.txt) might be trained as follows.  For older texts, it might also be useful to specify `-alternateSpellingReplacementPaths` or `-insertLongS true`, as shown here:

      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.TrainLanguageModel -mx7g -jar ocular-0.2-SNAPSHOT-with_dependencies.jar \
        -lmPath lm/trilingual.lmser \
        -textPath "spanish->texts/sp/,latin->texts/la/,nahuatl->texts/na/" \
        -alternateSpellingReplacementPaths "spanish->replace/spanish.txt,latin->replace/latin.txt,nahuatl->replace/nahuatl.txt" \
        -insertLongS true

  This program will work with any languages, and any number of languages; simply add an entry for every relevant language.  The set of languages chosen should match the set of languages found in the documents that are to be transcribed.

  More details on the various command-line options can be found below.


2. Initialize a font:

  Before a font can be trained from texts, a font model consisting of a "guess" for each character must be initialized based on the fonts on your computer.  Use `-fontPath` to specify where the initialized font should be written.  Since different languages use different character sets, a language model must be given in order for the system to know what characters to initialize (`-lmPath`).

      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.InitializeFont -mx7g -jar ocular-0.2-SNAPSHOT-with_dependencies.jar \
        -lmPath lm/trilingual.lmser \
        -fontPath font/trilingual/init.fontser


3. Train a font:

  To train a font, a set of document pages must be given (`-inputPath`), along with the paths to the language model and initial font model.  Use `-outputFontPath` to specify where the trained font model should be written, and `-outputPath` to specify where transcriptions and evaluation metrics should be written.  The path specified by `-inputPath` should point to a pdf or image file or directory or directory hierarchy of such files.  The value given by `-inputPath` will be searched recursively for non-`.txt` files; the transcriptions written to the `-outputPath` will maintain the same directory hierarchy.

      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.TranscribeOrTrainFont -mx7g -jar ocular-0.2-SNAPSHOT-with_dependencies.jar \
        -learnFont true \
        -initFontPath font/trilingual/init.fontser \
        -lmPath lm/trilingual.lmser \
        -inputPath sample_images/advertencias \
        -numDocs 10 \
        -outputFontPath font/advertencias/trained.fontser \
        -outputPath train_output
    
  If a gold standard transcription is available for a file, it should be written in a `.txt` file in the same directory as the corresponding image, and given the same filename (but with a different extension).  These files will be used to evaluate the accuracy of the transcription (during either training or testing).  For example:

      path/to/some/image_001.jpg      # document image
      path/to/some/image_001.txt      # corresponding transcription

  For extra speed, use `-emissionEngine OPENCL` if you have a Mac with a GPU, or `-emissionEngine CUDA` if you have Cuda installed.

  Many more command-line options can be found below.


4. Transcribe some pages:

  To transcribe pages, use the same instructions as above in #3 that were used to train a font, but leave `-learnFont` unspecified (or set it to `false`).  Additionally, `-initFontPath` should point to the newly-trained font model (instead of the "initial" font model used during font training).

      java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.TranscribeOrTrainFont -mx7g -jar ocular-0.2-SNAPSHOT-with_dependencies.jar \
        -inputPath sample_images/advertencias \
        -initFontPath font/advertencias/trained.fontser \
        -lmPath lm/trilingual.lmser \
        -outputPath transcribe_output 







## All Command-Line Options


### TrainLanguageModel

* `-lmPath`: Output Language Model file path. 
Required.

* `-textPath`: Path to the text files (or directory hierarchies) for training the LM.  For each entry, the entire directory will be recursively searched for any files that do not start with `.`.  For a multilingual (code-switching) model, give multiple comma-separated files with language names: `"english->texts/english/,spanish->texts/spanish/,french->texts/french/"`.  If spaces are used, be sure to wrap the whole string with "quotes".).
Required.

* `-languagePriors`: Prior probability of each language; ignore for uniform priors. Give multiple comma-separated language, prior pairs: `english->0.7,spanish->0.2,french->0.1`. If spaces are used, be sure to wrap the whole string with "quotes".  (Only relevant if multiple languages used.) 
Default: uniform priors

* `-pKeepSameLanguage`: Prior probability of sticking with the same language when moving between words in a code-switch model transition model.  (Only relevant if multiple languages used.) 
Default: 0.999999

* `-alternateSpellingReplacementPaths`: Paths to Alternate Spelling Replacement files. Give multiple comma-separated language, path pairs: `english->rules/en.txt,spanish->rules/sp.txt,french->rules/fr.txt`. If spaces are used, be sure to wrap the whole string with "quotes". Any languages for which no replacements are needed can be safely ignored. 
Default: no replacements

* `-insertLongS`: Use separate character type for long s.
Default: false

* `-removeDiacritics`: Remove diacritics? 
Default: false

* `-explicitCharacterSet`: A set of valid characters. If a character with a diacritic is found but not in this set, the diacritic will be dropped. Other excluded characters will simply be dropped. Ignore to allow all characters. *Not currently implemented*.
Default: ...

* `-maxLines`: Maximum number of lines to use from corpus.
Default: 1000000

* `-charN`: "LM character n-gram length."
Default: 6

* `-power`: exponent on LM scores.
Default: 4.0

* `-lmCharCount`: Number of characters to use for training the LM.  Use -1 to indicate that the full training data should be used.
Default: -1



### InitializeFont

* `-lmPath`: Path to the language model file (so that it knows which characters to create images for).
Required.

* `-fontPath`: Output font file path.
Required.

* `-numFontInitThreads`: Number of threads to use.
Deafult: 8

* `-templateMaxWidthFraction`: Max template width as fraction of text line height.
Default: 1.0

* `-templateMinWidthFraction`: Min template width as fraction of text line height.
Default: 0.0

* `-spaceMaxWidthFraction`: Max space template width as fraction of text line height.
Default: 1.0

* `-spaceMinWidthFraction`: Min space template width as fraction of text line height.
Default: 0.0



### TranscribeOrTrainFont

* `-inputPath`: Path of the directory that contains the input document images or pdfs. The entire directory will be recursively searched for any files that do not end in `.txt` (and that do not start with `.`).
Required.

* `-numDocs`: Number of documents to use. Ignore or use -1 to use all documents.
Default: -1

* `-lmPath`: Path to the language model file.
Required.

* `-initFontPath`: Path of the font initializer file.
Required.

* `-existingExtractionsPath`: If there are existing extractions, where to find them.  Ignore to perform new extractions.  *Not currently implemented*.
Default: null

* `-learnFont`: Whether to learn the font from the input documents and write the font to a file.
Default: false

* `-numEMIters`: Number of iterations of EM to use for font learning.
Default: 3

* `-outputPath`: Path of the directory that will contain output transcriptions and line extractions.
Required.

* `-lineExtractionOutputPath`: Path of the directory where the line-extraction images should be written.  If ignored, no images will be written.
Default: null

* `-outputFontPath`: Path to write the learned font file to. (Only if learnFont is set to true.)
Required if learnFont=true, otherwise ignored.

* `-outputLmPath`: Path to write the learned language model file to. (Only if learnFont is set to true.)
Default: null  (Don't write out the trained LM.)

* `-allowLanguageSwitchOnPunct`: A language model to be used to assign diacritics to the transcription output.
Default: true

* `-binarizeThreshold`: Quantile to use for pixel value thresholding. (High values mean more black pixels.)
Default: 0.12

* `-crop`: Crop pages?
Default: true

* `-markovVerticalOffset`: Use Markov chain to generate vertical offsets. (Slower, but more accurate. Turning on Markov offsets my require larger beam size for good results.)
Default: false

* `-beamSize`: Size of beam for Viterbi inference. (Usually in range 10-50. Increasing beam size can improve accuracy, but will reduce speed.)
Default: 10

* `-emissionEngine`: Engine to use for inner loop of emission cache computation. `DEFAULT`: Uses Java on CPU, which works on any machine but is the slowest method. `OPENCL`: Faster engine that uses either the CPU or integrated GPU (depending on processor) and requires OpenCL installation. `CUDA`: Fastest method, but requires a discrete NVIDIA GPU and CUDA installation.
Default: DEFAULT

* `-cudaDeviceID`: GPU ID when using CUDA emission engine.
Default: 0

* `-numMstepThreads`: Number of threads to use for LFBGS during m-step.
Default: 8

* `-numEmissionCacheThreads`: Number of threads to use during emission cache compuation. (Only has effect when emissionEngine is set to DEFAULT.)
Default: 8

* `-numDecodeThreads`: Number of threads to use for decoding. (Should be no smaller than decodeBatchSize.)
Default: 8

* `-decodeBatchSize`: Number of lines that compose a single decode batch. (Smaller batch size can reduce memory consumption.)
Default: 32

* `-paddingMinWidth`: Min horizontal padding between characters in pixels. (Best left at default value: 1.)
Default: 1

* `-paddingMaxWidth`: Max horizontal padding between characters in pixels (Best left at default value: 5.)
Default: 5



