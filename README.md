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

1. Initialize a font:

        edu.berkeley.cs.nlp.ocular.main.FontInitMain -fontPath font/init.fontser
    
2. Train a language model:

  Put some text in a file called `texts/test.txt`.  (Or download some off the web: http://www.gutenberg.org/cache/epub/2600/pg2600.txt).
    
      edu.berkeley.cs.nlp.ocular.main.LMTrainMain -lmPath lm/my_lm.lmser -textPath texts/test.txt -useLongS false
    
3. Train a font:

        edu.berkeley.cs.nlp.ocular.main.Main -learnFont true -initFontPath font/init.fontser -lmPath lm/my_lm.lmser -inputPath test_img -outputFontPath font/trained.fontser -outputPath train_output
    
  Probably need `-mx7g`.  For extra speed, use `-emissionEngine OPENCL` if you have a Mac, or `-emissionEngine CUDA` if you have Cuda installed.
    
4. Transcribe some pages:

        edu.berkeley.cs.nlp.ocular.main.Main -learnFont false -initFontPath font/trained.fontser -lmPath lm/my_lm.lmser -inputPath test_img -outputPath transcribe_output 
    

