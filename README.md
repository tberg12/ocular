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






## Running the full system

1. Train a language model:

  Put some text files in a folder called `texts/english/`.  (Or use, for example, `-textPath LICENSE.txt`).
    
      edu.berkeley.cs.nlp.ocular.main.LMTrainMain \
        -lmPath lm/my_lm.lmser \
        -textPath texts/english/ \
        -insertLongS false

2. Initialize a font:

        edu.berkeley.cs.nlp.ocular.main.FontInitMain \
          -lmPath lm/my_lm.lmser \
          -fontPath font/init.fontser

3. Train a font:

        edu.berkeley.cs.nlp.ocular.main.Main \
          -learnFont true \
          -initFontPath font/init.fontser \
          -lmPath lm/my_lm.lmser \
          -inputPath test_img/english \
          -outputFontPath font/trained.fontser \
          -outputPath train_output
    
  Probably need `-mx7g`.  For extra speed, use `-emissionEngine OPENCL` if you have a Mac, or `-emissionEngine CUDA` if you have Cuda installed.
    
4. Transcribe some pages:

        edu.berkeley.cs.nlp.ocular.main.Main \
          -inputPath test_img/english \
          -initFontPath font/trained.fontser \
          -lmPath lm/my_lm.lmser \
          -outputPath transcribe_output 


### Running the multilingual system, with a code-switching language model

1. Train a code-switching language model:

  Put some text in a folders called `texts/spanish/`, `texts/latin/`, and `texts/nahuatl/`.
    
      edu.berkeley.cs.nlp.ocular.main.CodeSwitchLMTrainMain \
        -lmPath lm/cs_lm.lmser \
        -textPaths "spanish->texts/spanish/,latin->texts/latin/,nahuatl->texts/nahuatl/" \
        -alternateSpellingReplacementPaths "spanish->replace/spanish.txt,latin->replace/latin.txt,nahuatl->replace/nahuatl.txt" \
        -insertLongS true

2. Initialize a font:

        edu.berkeley.cs.nlp.ocular.main.FontInitMain \
          -lmPath lm/cs_lm.lmser \
          -fontPath font/init_cs.fontser

3. Train a font:

        edu.berkeley.cs.nlp.ocular.main.MultilingualMain \
          -learnFont true \
          -inputPath test_img/multilingual \
          -numDocs 10 \
          -initFontPath font/cs_init.fontser \
          -initLmPath lm/cs_lm.lmser \
          -outputFontPath font/cs_trained.fontser \
          -outputLmPath lm/cs_trained.lmser \
          -outputPath cs_train_output
    
  Probably need `-mx7g`.  For extra speed, use `-emissionEngine OPENCL` if you have a Mac, or `-emissionEngine CUDA` if you have Cuda installed.
    
4. Transcribe some pages:

        edu.berkeley.cs.nlp.ocular.main.MultilingualMain \
          -inputPath test_img/multilingual \
          -initFontPath font/cs_trained.fontser \
          -initLmPath lm/cs_trained.lmser \
          -outputPath cs_transcribe_output 

    

