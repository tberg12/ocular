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



## Using Ocular

### Getting set up

There are three ways to use this repository:

1. Clone this repository, and compile the project into a jar:

        git clone https://github.com/tberg12/ocular.git
        cd ocular
        ./compile.sh

  This creates a jar file `ocular-0.2-SNAPSHOT-with_dependencies.jar` that can be run like:
  
        java -Done-jar.main.class=edu.berkeley.cs.nlp.ocular.main.Main -mx7g -jar ocular-0.2-SNAPSHOT-with_dependencies.jar [options...]

  This jar includes all the necessary dependencies, so you should be able to move it wherever you like, without the rest of the contents of this repository.

2. Clone this repository, and compile into a script:

        git clone https://github.com/tberg12/ocular.git
        cd ocular
        ./compile.sh
 
  This creates an executable script `target/start` that can be run like:
  
        export JAVA_OPTS="-mx7g"     # Increase the available memory
        target/start edu.berkeley.cs.nlp.ocular.main.Main [options...]

3. Use a dependency management system like Maven or SBT:

    * Repository location: `http://www.cs.utexas.edu/~dhg/maven-repository/snapshots`
    * Group ID: `edu.berkeley.cs.nlp`
    * Artifact ID: `ocular`
    * Version: `0.2-SNAPSHOT`
    
  For example, in SBT, you would include the following in your `build.sbt`:
  
      resolvers += "dhg snapshot repo" at "http://www.cs.utexas.edu/~dhg/maven-repository/snapshots"
      
      libraryDependencies += "edu.berkeley.cs.nlp" % "ocular" % "0.2-SNAPSHOT"


### Running the full system
**Note:** The following instructions allow you to run the original Ocular system as presented in Berg-Kirkpatrick et al. 2012 and 2013. This system uses a monolingual language model. For multilingual texts or documents written with high orthographic variation, consider using the multilingual system (see instructions below).

**Note:** The following instructions assume that you used "option 2" above to create an executable script.  If, instead, you would like to use "option 1" to create a jar, simply replace `target/start MAIN-CLASS` in each stage below with `java -Done-jar.main.class=MAIN-CLASS -mx7g -jar ocular-0.2-SNAPSHOT-with_dependencies.jar`.

**Note:** These instructions assume a general English corpus, but you can use any language or period-specific corpus of your choice.

1. Train a language model:

  Put some text files in a folder called `texts/english/`.  (For example, [download a book](http://www.gutenberg.org/cache/epub/2600/pg2600.txt)). 
    
      target/start edu.berkeley.cs.nlp.ocular.main.LMTrainMain \
        -lmPath lm/my_lm.lmser \
        -textPath texts/english/ \
        -insertLongS false

2. Initialize a font:

        target/start edu.berkeley.cs.nlp.ocular.main.FontInitMain \
          -lmPath lm/my_lm.lmser \
          -fontPath font/init.fontser

3. Train a font:

  To train using the pages that are in `test_img/english`, run:

        target/start edu.berkeley.cs.nlp.ocular.main.Main \
          -learnFont true \
          -initFontPath font/init.fontser \
          -lmPath lm/my_lm.lmser \
          -inputPath sample_images/english \
          -outputFontPath font/trained.fontser \
          -outputPath train_output
    
  For extra speed, use `-emissionEngine OPENCL` if you have a Mac with a GPU, or `-emissionEngine CUDA` if you have Cuda installed.
    
4. Transcribe some pages:

  To transcribe the pages that are in `test_img/english`, run:

        target/start edu.berkeley.cs.nlp.ocular.main.Main \
          -inputPath sample_images/english \
          -initFontPath font/trained.fontser \
          -lmPath lm/my_lm.lmser \
          -outputPath transcribe_output 


### Running the multilingual system, with a code-switching language model
**Note:** These instructions assume a trilingual corpus in Latin, Spanish, and Nahuatl, but the code is not specific to any language or any number of languages. Folder names and language data can be changed to match your project, and arguments can be modified for any number of languages following the pattern described below.

1. Train a code-switching language model:

  Put some text in a folders called `texts/spanish/`, `texts/latin/`, and `texts/nahuatl/`.  (For example, [Don Quijote](https://www.gutenberg.org/cache/epub/2000/pg2000.txt), [Meditationes de prima philosophia](https://www.gutenberg.org/cache/epub/23306/pg23306.txt), and [Ancient Nahuatl Poetry](https://www.gutenberg.org/cache/epub/12219/pg12219.txt)).
    
      target/start edu.berkeley.cs.nlp.ocular.main.CodeSwitchLMTrainMain \
        -lmPath lm/cs_lm.cslmser \
        -textPaths "spanish->texts/spanish/,latin->texts/latin/,nahuatl->texts/nahuatl/" \
        -alternateSpellingReplacementPaths "spanish->replace/spanish.txt,latin->replace/latin.txt,nahuatl->replace/nahuatl.txt" \
        -insertLongS true
        
2. Initialize a font:

        target/start edu.berkeley.cs.nlp.ocular.main.FontInitMain \
          -lmPath lm/cs_lm.cslmser \
          -fontPath font/cs_init.fontser

3. Train a font:

  To train using the pages that are in `test_img/multilingual`, run:

        target/start edu.berkeley.cs.nlp.ocular.main.MultilingualMain \
          -learnFont true \
          -inputPath sample_images/multilingual \
          -numDocs 10 \
          -initFontPath font/cs_init.fontser \
          -initLmPath lm/cs_lm.cslmser \
          -outputFontPath font/cs_trained.fontser \
          -outputLmPath lm/cs_trained.cslmser \
          -outputPath cs_train_output \
          -lineExtractionOutputPath cs_train_output
    
  For extra speed, use `-emissionEngine OPENCL` if you have a Mac with a GPU, or `-emissionEngine CUDA` if you have Cuda installed.
    
4. Transcribe some pages:

  To train using the pages that are in `test_img/multilingual`, run:

        target/start edu.berkeley.cs.nlp.ocular.main.MultilingualMain \
          -inputPath sample_images/multilingual \
          -initFontPath font/cs_trained.fontser \
          -initLmPath lm/cs_trained.cslmser \
          -outputPath cs_transcribe_output \
          -lineExtractionOutputPath cs_transcribe_output

    


## All Command-Line Options

### LMTrainMain

* `-lmPath`: Output LM file path.
Required.

* `-textPath`: Input corpus path.
Required.

* `-insertLongS`: Use separate character type for long s.
Default: false

* `-removeDiacritics`: Remove diacritics?
Default: false

* `-explicitCharacterSet`: A set of valid characters. If a character with a diacritic is found but not in this set, the diacritic will be dropped. Other excluded characters will simply be dropped. Ignore to allow all characters. *Not currently implemented*.
Default: ...

* `-maxLines`: Maximum number of lines to use from corpus.
Default: 1000000

* `-charN`: LM character n-gram length.
Default: 6

* `-power`: Exponent on LM scores.
Default: 4.0



### FontInitMain

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




### Main

* `-inputPath`: Path of the directory that contains the input document images.
Required.

* `-lmPath`: Path to the language model file.
Required.

* `-initFontPath`: Path of the font initializer file.
Required.

* `-learnFont`: Whether to learn the font from the input documents and write the font to a file.
Default: false

* `-outputPath`: Path of the directory that will contain output transcriptions and line extractions.
Required.

* `-outputFontPath`: Path to write the learned font file to. (Only if learnFont is set to true.)
Required.

* `-numEMIters`: Number of iterations of EM to use for font learning.
Default: 4

* `-binarizeThreshold`: Quantile to use for pixel value thresholding. (High values mean more black pixels.)
Default: 0.12

* `-markovVerticalOffset`: Use Markov chain to generate vertical offsets. (Slower, but more accurate. Turning on Markov offsets my require larger beam size for good results.)
Default: true

* `-beamSize`: Size of beam for viterbi inference. (Usually in range 10-50. Increasing beam size can improve accuracy, but will reduce speed.)
Default: 10

* `-emissionEngine`: Engine to use for inner loop of emission cache computation. `DEFAULT`: Uses Java on CPU, which works on any machine but is the slowest method. `OPENCL`: Faster engine that uses either the CPU or integrated GPU (depending on processor) and requires OpenCL installation. `CUDA`: Fastest method, but requires a discrete NVIDIA GPU and CUDA installation.
Default: DEFAULT

* `-cudaDeviceID`: GPU ID when using CUDA emission engine.
Default: 0

* `-numMstepThreads`: Number of threads to use for LFBGS during m-step.
Default: 8

* `-numEmissionCacheThreads`: Number of threads to use during emission cache compuation. (Only has affect when emissionEngine is set to DEFAULT.)
Default: 8

* `-numDecodeThreads`: Number of threads to use for decoding. (Should be no smaller than decodeBatchSize.)
Default: 8

* `-decodeBatchSize`: Number of lines that compose a single decode batch. (Smaller batch size can reduce memory consumption.)
Default: 32

* `-paddingMinWidth`: Min horizontal padding between characters in pixels. (Best left at default value: 1.)
Default: 1

* `-paddingMaxWidth`: Max horizontal padding between characters in pixels (Best left at default value: 5.)
Default: 5




### CodeSwitchLMTrainMain

* `-lmPath`: Output Language Model file path. 
Required.

* `-textPaths`: Path to the text files (or directory hierarchies) for training the LM. (For multiple paths for multilingual (code-switching) support, give multiple comma-separated files with language names: `english->lms/english/,spanish->lms/spanish/,french->lms/french/`.  If spaces are used, be sure to wrap the whole string with "quotes".  For each entry, the entire directory will be recursively searched for any files that do not start with `.`.
Required.

* `-languagePriors`: Prior probability of each language; ignore for uniform priors. Give multiple comma-separated language, prior pairs: `english->0.7,spanish->0.2,french->0.1`. If spaces are used, be sure to wrap the whole string with "quotes". 
Default: null  (uniform priors)

* `-pKeepSameLanguage`: Prior probability of sticking with the same language when moving between words in a code-switch model transition model.  (For use with codeSwitch.) 
Default: 0.999999

* `-alternateSpellingReplacementPaths`: Paths to Alternate Spelling Replacement files. Give multiple comma-separated language, path pairs: `english->rules/en.txt,spanish->rules/sp.txt,french->rules/fr.txt`. If spaces are used, be sure to wrap the whole string with "quotes". Any languages for which no replacements are needed can be safely ignored. 
Default: null  (no replacements)

* `-insertLongS`: Use separate character type for long s.
Default: false

* `-removeDiacritics`: Remove diacritics? 
Default: false.

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




### MultilingualMain

* `-inputPath`: Path of the directory that contains the input document images. The entire directory will be recursively searched for any files that do not end in `.txt` (and that do not start with `.`).
Required.

* `-numDocs`: Number of documents to use. Ignore or use -1 to use all documents.
Default: -1

* `-initLmPath`: Path to the language model file.
Required.

* `-initFontPath`: Path of the font initializer file.
Required.

* `-existingExtractionsPath`: If there are existing extractions (from a previous `-lineExtractionOutputPath`), where to find them.  Ignore to perform new extractions.  *Not currently implemented*.
Default: null

* `-learnFont`: Whether to learn the font from the input documents and write the font to a file.
Default: false

* `-numEMIters`: Number of iterations of EM to use for font learning.
Default: 4

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
Default: false

* `-markovVerticalOffset`: Use Markov chain to generate vertical offsets. (Slower, but more accurate. Turning on Markov offsets my require larger beam size for good results.)
Default: false

* `-beamSize`: Size of beam for viterbi inference. (Usually in range 10-50. Increasing beam size can improve accuracy, but will reduce speed.)
Default: 10

* `-emissionEngine`: Engine to use for inner loop of emission cache computation. `DEFAULT`: Uses Java on CPU, which works on any machine but is the slowest method. `OPENCL`: Faster engine that uses either the CPU or integrated GPU (depending on processor) and requires OpenCL installation. `CUDA`: Fastest method, but requires a discrete NVIDIA GPU and CUDA installation.
Default: DEFAULT

* `-cudaDeviceID`: GPU ID when using CUDA emission engine.
Default: 0

* `-numMstepThreads`: Number of threads to use for LFBGS during m-step.
Default: 8

* `-numEmissionCacheThreads`: Number of threads to use during emission cache compuation. (Only has affect when emissionEngine is set to DEFAULT.)
Default: 8

* `-numDecodeThreads`: Number of threads to use for decoding. (Should be no smaller than decodeBatchSize.)
Default: 8

* `-decodeBatchSize`: Number of lines that compose a single decode batch. (Smaller batch size can reduce memory consumption.)
Default: 32

* `-paddingMinWidth`: Min horizontal padding between characters in pixels. (Best left at default value: 1.)
Default: 1

* `-paddingMaxWidth`: Max horizontal padding between characters in pixels (Best left at default value: 5.)
Default: 5



