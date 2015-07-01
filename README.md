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

**Note:** The following instructions assume that you used "option 2" above to create an executable script.  If, instead, you would like to use "option 1" to create a jar, simply replace `target/start MAIN-CLASS` in each stage below with `java -Done-jar.main.class=MAIN-CLASS -jar ocular-0.2-SNAPSHOT-with_dependencies.jar`.

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

        target/start edu.berkeley.cs.nlp.ocular.main.Main \
          -learnFont true \
          -initFontPath font/init.fontser \
          -lmPath lm/my_lm.lmser \
          -inputPath test_img/english \
          -outputFontPath font/trained.fontser \
          -outputPath train_output
    
  For extra speed, use `-emissionEngine OPENCL` if you have a Mac, or `-emissionEngine CUDA` if you have Cuda installed.
    
4. Transcribe some pages:

        target/start edu.berkeley.cs.nlp.ocular.main.Main \
          -inputPath test_img/english \
          -initFontPath font/trained.fontser \
          -lmPath lm/my_lm.lmser \
          -outputPath transcribe_output 


### Running the multilingual system, with a code-switching language model

1. Train a code-switching language model:

  Put some text in a folders called `texts/spanish/`, `texts/latin/`, and `texts/nahuatl/`.  (For example, [Don Quijote](https://www.gutenberg.org/cache/epub/2000/pg2000.txt), [Meditationes de prima philosophia](https://www.gutenberg.org/cache/epub/23306/pg23306.txt), and [Ancient Nahuatl Poetry](https://www.gutenberg.org/cache/epub/12219/pg12219.txt)).
    
      target/start edu.berkeley.cs.nlp.ocular.main.CodeSwitchLMTrainMain \
        -lmPath lm/cs_lm.lmser \
        -textPaths "spanish->texts/spanish/,latin->texts/latin/,nahuatl->texts/nahuatl/" \
        -alternateSpellingReplacementPaths "spanish->replace/spanish.txt,latin->replace/latin.txt,nahuatl->replace/nahuatl.txt" \
        -insertLongS true
        
2. Initialize a font:

        target/start edu.berkeley.cs.nlp.ocular.main.FontInitMain \
          -lmPath lm/cs_lm.lmser \
          -fontPath font/cs_init.fontser

3. Train a font:

        target/start edu.berkeley.cs.nlp.ocular.main.MultilingualMain \
          -learnFont true \
          -inputPath test_img/multilingual \
          -numDocs 10 \
          -initFontPath font/cs_init.fontser \
          -initLmPath lm/cs_lm.lmser \
          -outputFontPath font/cs_trained.fontser \
          -outputLmPath lm/cs_trained.lmser \
          -outputPath cs_train_output \
          -lineExtractionOutputPath cs_train_output
    
  Probably need `-mx7g`.  For extra speed, use `-emissionEngine OPENCL` if you have a Mac, or `-emissionEngine CUDA` if you have Cuda installed.
    
4. Transcribe some pages:

        target/start edu.berkeley.cs.nlp.ocular.main.MultilingualMain \
          -inputPath test_img/multilingual \
          -initFontPath font/cs_trained.fontser \
          -initLmPath lm/cs_trained.lmser \
          -outputPath cs_transcribe_output \
          -lineExtractionOutputPath cs_transcribe_output

    


