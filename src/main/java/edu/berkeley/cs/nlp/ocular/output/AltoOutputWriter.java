package edu.berkeley.cs.nlp.ocular.output;

import static org.apache.commons.lang3.StringEscapeUtils.escapeHtml3;
import indexer.Indexer;

import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.Iterator;
import java.util.List;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import util.Iterators;
import edu.berkeley.cs.nlp.ocular.data.Document;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar.GlyphType;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.util.StringHelper;
import fileio.f;

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

	public void write(int numLines, List<TransitionState>[] viterbiTransStates, Document doc, String outputFilenameBase, String inputDocPath, List<Integer>[] viterbiWidths, List<String> commandLineArgs, boolean outputNormalized) {
		String altoOutputFilename = outputFilenameBase + (outputNormalized ? "_norm" : "_dipl") + ".alto.xml";

		SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd'T'hh:mm:ss");
		String imgFilename = doc.baseName();
		
		StringBuffer outputBuffer = new StringBuffer();
		outputBuffer.append("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
		outputBuffer.append("<alto xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" xmlns=\"http://www.loc.gov/standards/alto/ns-v3#\" xsi:schemaLocation=\"http://www.loc.gov/standards/alto/ns-v3# http://www.loc.gov/standards/alto/v3/alto.xsd\" xmlns:emop=\"http://emop.tamu.edu\">\n");
		outputBuffer.append("  <Description>\n");
		outputBuffer.append("    <MeasurementUnit>pixel</MeasurementUnit>\n");
		outputBuffer.append("    <sourceImageInformation>\n");
		outputBuffer.append("      <fileName>"+imagePathToFilename(imgFilename)+"</fileName>\n");  //gives filename with extension
		outputBuffer.append("    </sourceImageInformation>\n");
		outputBuffer.append("    <OCRProcessing ID=\"Ocular0.0.3\">\n"); 
		outputBuffer.append("      <preProcessingStep></preProcessingStep>\n"); 
		outputBuffer.append("      <ocrProcessingStep>\n");
		outputBuffer.append("		 <processingDateTime>"+formatter.format(new Date())+"</processingDateTime>\n");
//			for (StringBuffer s: args) {
//				StringBuffer arguments = new StringBuffer();
//				arguments.append(s);
//			};
		outputBuffer.append("      	 <processingStepSettings>"+StringHelper.join(commandLineArgs, " ")+"</processingStepSettings>\n");
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
		outputBuffer.append("    <Page ID=\""+imageFilenameToId(imgFilename)+"\"  PHYSICAL_IMG_NR=\""+imageFilenameToIdNumber(imgFilename)+"\" >\n");
		outputBuffer.append("      <PrintSpace>\n");
		outputBuffer.append("        <TextBlock ID=\"par_1\">\n");
		
		boolean inWord = false; // (as opposed to a space)
		int wordIndex = 0;
		for (int line = 0; line < numLines; ++line) {
			boolean beginningOfLine = true;
			
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
									diplomaticTranscriptionBuffer.append(Charset.unescapeChar(charIndexer.getObject(wts.getGlyphChar().templateCharIndex))); //w/ normalized ocular, we'll want to preserve things like "shorthand" or whatever.
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
										String s = Charset.unescapeChar(charIndexer.getObject(wts.getLmCharIndex()));
										if (s.equals(Charset.LONG_S)) s = "s"; // don't use long-s in "normalized" transcriptions
										normalizedTranscriptionBuffer.append(s);
									}
								}
							}
							String diplomaticTranscription = diplomaticTranscriptionBuffer.toString().trim();
							String normalizedTranscription = normalizedTranscriptionBuffer.toString().trim(); //Use this to add in the norm
							if (!diplomaticTranscription.isEmpty()) {
								outputBuffer.append("      <String ID=\"word_"+wordIndex+"\" WIDTH=\""+wordWidth+"\" CONTENT=\""+escapeHtml3(outputNormalized ? normalizedTranscription : diplomaticTranscription)+"\" LANG=\""+language+"\"");
								if (!normalizedTranscription.equals(diplomaticTranscription)) {
									outputBuffer.append("> \n");
									if (outputNormalized) {
										outputBuffer.append("          <ALTERNATIVE PURPOSE=\"Diplomatic\">"+escapeHtml3(normalizedTranscription)+"</ALTERNATIVE>\n");
									}
									else {
										outputBuffer.append("          <ALTERNATIVE PURPOSE=\"Normalization\">"+escapeHtml3(diplomaticTranscription)+"</ALTERNATIVE>\n");	
									}
									outputBuffer.append("      </String>\n");
								}
								else {
									outputBuffer.append("/> \n");
								}
								beginningOfLine = false;
								wordIndex = wordIndex+1;
							}
						}
					}
					else { // in space TODO: This cannot happen at the beginning of a line.
						if (!beginningOfLine) {
							if (wordWidth > 0) {
								outputBuffer.append("      <SP WIDTH=\""+wordWidth+"\"/>\n");
							}
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
	
    private String imageFilenameToId(String imageFilename) { //pl_blac_012_00013-800.jpg
        String pattern = "(pl_[a-z]+_\\d+_\\d+).*";
        Pattern r = Pattern.compile(pattern);
        Matcher m = r.matcher(imageFilename);
        if (m.find()) {
            return m.group(1);
        } else {
            return "Error: page ID unknown";
        }
	}
    private String imageFilenameToIdNumber(String imageFilename) { //pl_blac_012_00013-800.jpg
        String pattern = "pl_[a-z]+_\\d+_(\\d+).*";
        Pattern r = Pattern.compile(pattern);
        Matcher m = r.matcher(imageFilename);
        if (m.find()) {
            return m.group(1);
        } else {
            return "Error: ID Number unknown";
        }
	}
    
    private String imagePathToFilename(String imageFilename) { //pl_blac_012_00013-800.jpg
        String pattern = ".*(pl_[a-z]+_\\d+_\\d+.*)";
        Pattern r = Pattern.compile(pattern);
        Matcher m = r.matcher(imageFilename);
        if (m.find()) {
            return m.group(1);
        } else {
            return "Error: filename unknown";
        }
	}
	
}
