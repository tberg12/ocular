# ocular

Ocular is a state-of-the-art historical OCR system.



## Running the full system

1. Initialize a font:

        edu.berkeley.cs.nlp.ocular.main.FontInitMain -fontPath font/init.fontser
    
2. Train a language model:

  Put some text in a file called `texts/test.txt`.  (Or download some off the web: http://www.gutenberg.org/cache/epub/2600/pg2600.txt).  It must contain every character used by the system: `0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz &.,;:"'!?()-` (but you can just copy and paste that string in).
    
      edu.berkeley.cs.nlp.ocular.main.LMTrainMain -lmPath lm/my_lm.lmser -textPath texts/test.txt -useLongS false
    
3. Train a font:

        edu.berkeley.cs.nlp.ocular.main.Main -learnFont true -initFontPath font/init.fontser -lmPath lm/my_lm.lmser -inputPath test_img -outputFontPath font/trained.fontser -outputPath train_output
    
  Probably need `-mx7g`.  For extra speed, use `-emissionEngine OPENCL` if you have a Mac, or `-emissionEngine CUDA` if you have Cuda installed.
    
4. Transcribe some pages:

        edu.berkeley.cs.nlp.ocular.main.Main -learnFont false -initFontPath font/trained.fontser -lmPath lm/my_lm.lmser -inputPath test_img -outputPath transcribe_output 
    

