package edu.berkeley.cs.nlp.ocular.output;

import java.io.File;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Iterator;
import java.util.List;

import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar.GlyphType;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import fileio.f;
import indexer.Indexer;
import util.Iterators;

/**
 * @author Hannah Alpert-Abrams (halperta@gmail.com)
 */
public class AltoOutputWriter {
	
	private Indexer<String> charIndexer;
	private Indexer<String> langIndexer;
	private int spaceCharIndex;
	private int hyphenCharIndex;
	
	public AltoOutputWriter(Indexer<String> charIndexer, Indexer<String> langIndexer) {
		this.charIndexer = charIndexer;
		this.langIndexer = langIndexer;
		
		this.spaceCharIndex = charIndexer.getIndex(Charset.SPACE);
		this.hyphenCharIndex = charIndexer.getIndex(Charset.HYPHEN);
	}

	public void write(int numLines, List<TransitionState>[] viterbiTransStates, String inputDocPath, String outputFilenameBase, List<Integer>[] viterbiWidths) {
		String altoOutputFilename = outputFilenameBase + ".alto.xml";
		
		StringBuffer outputBuffer = new StringBuffer();
		outputBuffer.append("<alto xmlns=\"http://schema.ccs-gmbh.com/ALTO\" xmlns:emop=\"http://emop.tamu.edu\">\n");
		outputBuffer.append("  <Description>\n");
		outputBuffer.append("    <MeasurementUnit>pixel</MeasurementUnit>\n");
		outputBuffer.append("    <sourceImageInformation>\n");
		outputBuffer.append("      <filename>"+(new File(inputDocPath).getName())+"</filename>\n");
		outputBuffer.append("    </sourceImageInformation>\n");
		outputBuffer.append("    <OCRProcessing>\n"); //IDK how we should "ID" this.
		outputBuffer.append("      <preProcessingStep></preProcessingStep>\n"); 
		outputBuffer.append("      <ocrProcessingStep>\n");
		outputBuffer.append("		 <processingDateTime>"+Calendar.getInstance()+"</processingDateTime>\n"); //not really working...
//			for (StringBuffer s: args) {
//				StringBuffer arguments = new StringBuffer();
//				arguments.append(s);
//			};
//			outputBuffer.append("      	 <processingStepSettings>"+arguments+"</processingStepSettings>\n");//want to print "args" here ...
		outputBuffer.append("        <processingSoftware>\n");
		outputBuffer.append("          <softwareCreator>Taylor Berg-Kirkpatrick, Greg Durrett, Dan Klein, Dan Garrette, Hannah Alpert-Abrams</softwareCreator>\n");
		outputBuffer.append("          <softwareName>Ocular</softwareName>\n");
		outputBuffer.append("          <softwareVersion>0.0.3</softwareVersion>\n");
		outputBuffer.append("        </processingSoftware>\n");
		outputBuffer.append("       </ocrProcessingStep>\n");
//			outputBuffer.append("      <postProcessingStep>\n");
//			outputBuffer.append("        <processingSoftware>\n");
//			outputBuffer.append("          <softwareCreator>\n");
//			outputBuffer.append("            Illinois Informatics Institute, University of Illinois at Urbana-Champaign http://www.informatics.illinois.edu\n");
//			outputBuffer.append("          </softwareCreator>\n");
//			outputBuffer.append("          <softwareName>PageCorrector</softwareName>\n");
//			outputBuffer.append("          <softwareVersion>1.10.0-SNAPSHOT</softwareVersion>\n");
//			outputBuffer.append("        </processingSoftware>\n");
//			outputBuffer.append("      </postProcessingStep>\n");
		outputBuffer.append("    </OCRProcessing>\n");
		outputBuffer.append("  </Description>\n");
		outputBuffer.append("  <Layout>\n");
		outputBuffer.append("    <Page ID=\"page_1\">\n");
		outputBuffer.append("      <PrintSpace>\n");
		outputBuffer.append("        <TextBlock ID=\"par_1\">\n");
		
		boolean inWord = false; // (as opposed to a space)
		int wordIndex = 0;
		for (int line = 0; line < numLines; ++line) {
			outputBuffer.append("    <TextLine ID=\"line_"+(line+1)+"\">\n"); //Opening <TextLine>, assigning ID.
			@SuppressWarnings("unchecked")
			Iterator<TransitionState> tsIterator = Iterators.concat(viterbiTransStates[line].iterator(), Iterators.<TransitionState>oneItemIterator(null));
			Iterator<Integer> widthsIterator = viterbiWidths[line].iterator();
			
			List<TransitionState> wordBuffer = new ArrayList<TransitionState>(); 
			int wordWidth = 0;
			while (tsIterator.hasNext()) {
				TransitionState ts = tsIterator.next();
				boolean isSpace = ts != null ? ts.getLmCharIndex() == spaceCharIndex && ts.getGlyphChar().templateCharIndex == spaceCharIndex : true;
				boolean isPunct = ts != null ? ts.getLmCharIndex() != hyphenCharIndex && Charset.isPunctuationChar(charIndexer.getObject(ts.getLmCharIndex())) : false;
				boolean endOfSpan = (isSpace == inWord) || isPunct || !tsIterator.hasNext(); // end of word, contiguous space sequence, or line
				
				if (endOfSpan) { // if we're at a transition point between spans, we need to write out the complete span's information
					if (inWord) { // if we're completing a word (as opposed to a sequence of spaces)
						if (!wordBuffer.isEmpty()) { // if there's wordiness to print out (hopefully this will always be true if we get to this point)
							int languageIndex = wordBuffer.get(0).getLanguageIndex();
							String language = languageIndex >= 0 ? langIndexer.getObject(languageIndex) : "None";
							StringBuffer diplomaticTranscriptionBuffer = new StringBuffer();
							StringBuffer normalizedTranscriptionBuffer = new StringBuffer();
							for (TransitionState wts : wordBuffer) {
								if (!wts.getGlyphChar().isElided()) {
									diplomaticTranscriptionBuffer.append(Charset.unescapeCharPrecomposedOnly(charIndexer.getObject(wts.getGlyphChar().templateCharIndex))); //w/ normalized ocular, we'll want to preserve things like "shorthand" or whatever.
								}
								if (wts.getGlyphChar().glyphType != GlyphType.DOUBLED) { // the first in a pair of doubled characters isn't part of the language model transcription
									switch(wts.getType()) {
									case RMRGN_HPHN_INIT:
										normalizedTranscriptionBuffer.append(Charset.HYPHEN);
										break;
									case RMRGN_HPHN:
									case LMRGN_HPHN:
										break;
									case LMRGN:
									case RMRGN:
										normalizedTranscriptionBuffer.append(Charset.SPACE);
										break;
									case TMPL:
										String s = Charset.unescapeCharPrecomposedOnly(charIndexer.getObject(wts.getLmCharIndex()));
										if (s.equals(Charset.LONG_S)) s = "s"; // don't use long-s in "normalized" transcriptions
										normalizedTranscriptionBuffer.append(s);
									}
								}
							}
							String diplomaticTranscription = diplomaticTranscriptionBuffer.toString().trim();
							String normalizedTranscription = normalizedTranscriptionBuffer.toString().trim(); //Use this to add in the norm
							if (!diplomaticTranscription.isEmpty()) {
								outputBuffer.append("      <String ID=\"word_"+wordIndex+"\" WIDTH=\""+wordWidth+"\" CONTENT=\""+diplomaticTranscription+"\" Language=\""+language+"\"> \n");
								outputBuffer.append("          <ALTERNATIVE>"+normalizedTranscription+"</ALTERNATIVE>\n");
								outputBuffer.append("      </String>\n");
								wordIndex = wordIndex+1;
							}
						}
					}
					else { // in space
						if (wordWidth > 0) {
							outputBuffer.append("      <SP WIDTH=\""+wordWidth+"\"/>\n");
						}
					}
					
					// get read to start a new span
					wordBuffer.clear(); 
					wordWidth = 0;
					inWord = !isSpace;
				}
				
				// add the current state into the (existing or freshly-cleared) span buffer
				wordBuffer.add(ts);
				wordWidth += (ts != null ? widthsIterator.next() : 0);
			}			
			
			outputBuffer.append("    </TextLine>\n");
		}
		outputBuffer.append("</TextBlock>\n");
		outputBuffer.append("</PrintSpace>\n");
		outputBuffer.append("</Page>\n");
		outputBuffer.append("</Layout>\n");
		outputBuffer.append("</alto>\n");
		String outputString = outputBuffer.toString();

		System.out.println("Writing alto output to " + altoOutputFilename);
		f.writeString(altoOutputFilename, outputString);
	}
	
}
