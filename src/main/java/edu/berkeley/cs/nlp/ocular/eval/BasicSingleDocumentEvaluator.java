package edu.berkeley.cs.nlp.ocular.eval;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.HYPHEN;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;

import java.io.File;
import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.Iterator;
import java.util.List;
import java.util.Map;

import edu.berkeley.cs.nlp.ocular.data.FileUtil;
import edu.berkeley.cs.nlp.ocular.data.ImageLoader.Document;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.eval.Evaluator.EvalSuffStats;
import edu.berkeley.cs.nlp.ocular.model.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.sub.GlyphChar;
import edu.berkeley.cs.nlp.ocular.sub.GlyphChar.GlyphType;
import edu.berkeley.cs.nlp.ocular.util.FileHelper;
import edu.berkeley.cs.nlp.ocular.util.StringHelper;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import fileio.f;
import indexer.Indexer;
import util.Iterators;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class BasicSingleDocumentEvaluator implements SingleDocumentEvaluator {
	private Indexer<String> charIndexer;
	private Indexer<String> langIndexer;
	private boolean allowGlyphSubstitution;
	private boolean charIncludesDiacritic;
	
	private int spaceCharIndex;
	private int hyphenCharIndex;
	
	public BasicSingleDocumentEvaluator(Indexer<String> charIndexer, Indexer<String> langIndexer, boolean allowGlyphSubstitution, boolean charIncludesDiacritic) {
		this.charIndexer = charIndexer;
		this.langIndexer = langIndexer;
		this.allowGlyphSubstitution = allowGlyphSubstitution;
		this.charIncludesDiacritic = charIncludesDiacritic;
		
		this.spaceCharIndex = charIndexer.getIndex(Charset.SPACE);
		this.hyphenCharIndex = charIndexer.getIndex(Charset.HYPHEN);
	}

	public void printTranscriptionWithEvaluation(int iter, int batchId,
			Document doc,
			TransitionState[][] decodeStates, int[][] decodeWidths,
			String inputDocPath, String outputPath,
			List<Tuple2<String, Map<String, EvalSuffStats>>> allEvals,
			List<Tuple2<String, Map<String, EvalSuffStats>>> allLmEvals) {
		String[][] text = doc.loadLineText();
		List<String> goldLmChars = doc.loadLmText();
		
		//
		// Make sure the decoded states and the text have the same number of lines (numLines)
		//
		int numLines = decodeStates.length;
		if (text != null && text.length > numLines) numLines = text.length; // in case gold and viterbi have different line counts
		
		if (text != null && text.length < numLines) {
			String[][] newText = new String[numLines][];
			for (int line = 0; line < numLines; ++line) {
				if (line < text.length)
					newText[line] = text[line];
				else
					newText[line] = new String[0];
			}
			text = newText;
		}
		if (decodeStates.length < numLines) {
			TransitionState[][] newDecodeStates = new TransitionState[numLines][];
			for (int line = 0; line < numLines; ++line) {
				if (line < decodeStates.length)
					newDecodeStates[line] = decodeStates[line];
				else
					newDecodeStates[line] = new TransitionState[0];
			}
			decodeStates = newDecodeStates;
		}

		//
		// Get the model output
		//
		@SuppressWarnings("unchecked")
		List<String>[] viterbiChars = new List[numLines];
		List<String> viterbiLmChars = new ArrayList<String>();
		@SuppressWarnings("unchecked")
		List<TransitionState>[] viterbiTransStates = new List[numLines];
		@SuppressWarnings("unchecked")
		List<Integer>[] viterbiWidths = new List[numLines];
		for (int line = 0; line < numLines; ++line) {
			viterbiChars[line] = new ArrayList<String>();
			viterbiTransStates[line] = new ArrayList<TransitionState>();
			viterbiWidths[line] = new ArrayList<Integer>();
			for (int i = 0; i < decodeStates[line].length; ++i) {
				TransitionState ts = decodeStates[line][i];
				int c = ts.getGlyphChar().templateCharIndex;
				if (viterbiChars[line].isEmpty() || !(HYPHEN.equals(viterbiChars[line].get(viterbiChars[line].size() - 1)) && HYPHEN.equals(charIndexer.getObject(c)))) {
					if (!ts.getGlyphChar().isElided()) {
						viterbiChars[line].add(charIndexer.getObject(c));
					}
					
					if (ts.getGlyphChar().glyphType != GlyphType.DOUBLED) { // the first in a pair of doubled characters isn't part of the language model transcription
						switch(ts.getType()) {
							case RMRGN_HPHN_INIT:
							case RMRGN_HPHN:
							case LMRGN_HPHN:
								break;
							case LMRGN:
							case RMRGN:
								viterbiLmChars.add(" ");
								break;
							case TMPL:
								String s = charIndexer.getObject(ts.getLmCharIndex());
								if (s.equals(Charset.LONG_S)) s = "s"; // don't use long-s in "normalized" transcriptions
								viterbiLmChars.add(s);
						}
					}
					
					viterbiTransStates[line].add(ts);
					viterbiWidths[line].add(decodeWidths[line][i]);
				}
			}
		}
		
		System.out.println("Viterbi LM Chars: " + StringHelper.join(viterbiLmChars));

		String fileParent = FileUtil.removeCommonPathPrefixOfParents(new File(inputDocPath), new File(doc.baseName()))._2;
		String preext = FileUtil.withoutExtension(new File(doc.baseName()).getName());
		String outputFilenameBase = outputPath + "/" + fileParent + "/" + preext;
		if (iter > 0) outputFilenameBase += "_iter-" + iter;
		if (batchId > 0) outputFilenameBase += "_batch-" + batchId;
		
		String transcriptionOutputFilename = outputFilenameBase + "_transcription.txt";
		String transcriptionWithSubsOutputFilename = outputFilenameBase + "_transcription_withSubs.txt";
		String transcriptionWithWidthsOutputFilename = outputFilenameBase + "_transcription_withWidths.txt";
		String goldComparisonOutputFilename = outputFilenameBase + "_vsGold.txt";
		String goldComparisonWithSubsOutputFilename = outputFilenameBase + "_vsGold_withSubs.txt";
		String goldLmComparisonOutputFilename = outputFilenameBase + "_lm_vsGold.txt";
		String htmlOutputFilename = outputFilenameBase + ".html";
		String altoOutputFilename = outputFilenameBase + ".alto.xml";
		new File(transcriptionOutputFilename).getParentFile().mkdirs();
		
		//
		// Plain transcription output
		//
		{
		System.out.println("Writing transcription output to " + transcriptionOutputFilename);
		StringBuffer transcriptionOutputBuffer = new StringBuffer();
		for (int line = 0; line < numLines; ++line) {
			transcriptionOutputBuffer.append(StringHelper.join(viterbiChars[line], "") + "\n");
		}
		//System.out.println(transcriptionOutputBuffer.toString() + "\n\n");
		FileHelper.writeString(transcriptionOutputFilename, transcriptionOutputBuffer.toString());
		}

		//
		// Transcription output with substitutions
		//
		List<String> transcriptionWithSubsOutputLines = new ArrayList<String>();
		if (allowGlyphSubstitution) {
		System.out.println("Transcription with substitutions");
		for (int line = 0; line < numLines; ++line) {
			StringBuilder lineBuffer = new StringBuilder();
			for (TransitionState ts : viterbiTransStates[line]) {
				int lmChar = ts.getLmCharIndex();
				GlyphChar glyph = ts.getGlyphChar();
				int glyphChar = glyph.templateCharIndex;
				String sglyphChar = Charset.unescapeChar(charIndexer.getObject(glyphChar));
				if (glyph.glyphType == GlyphType.DOUBLED) {
					lineBuffer.append("[2x");
					lineBuffer.append(Charset.unescapeChar(charIndexer.getObject(lmChar)));
					if (lmChar != glyphChar || glyph.glyphType != GlyphType.NORMAL_CHAR) {
						lineBuffer.append("/" + (glyph.isElided() ? "" : sglyphChar) + "]");
					}
					lineBuffer.append("]");
				}
				else if (lmChar != glyphChar || glyph.glyphType != GlyphType.NORMAL_CHAR) {
					lineBuffer.append("[" + Charset.unescapeChar(charIndexer.getObject(lmChar)) + "/" + (glyph.isElided() ? "" : sglyphChar) + "]");
				}
				else {
					lineBuffer.append(sglyphChar);
				}
			}
			transcriptionWithSubsOutputLines.add(lineBuffer.toString() + "\n");
		}
		String transcriptionWithSubsOutputBuffer = StringHelper.join(transcriptionWithSubsOutputLines, "");
		System.out.println(transcriptionWithSubsOutputBuffer.toString() + "\n\n");
		FileHelper.writeString(transcriptionWithSubsOutputFilename, transcriptionWithSubsOutputBuffer.toString());
		}

		//
		// Transcription with widths
		//
		if (allowGlyphSubstitution) {
		System.out.println("Transcription with widths");
		StringBuffer transcriptionWithWidthsOutputBuffer = new StringBuffer();
		for (int line = 0; line < numLines; ++line) {
			transcriptionWithWidthsOutputBuffer.append(transcriptionWithSubsOutputLines.get(line));
			for (int i = 0; i < viterbiTransStates[line].size(); ++i) {
				TransitionState ts = viterbiTransStates[line].get(i);
				int w = viterbiWidths[line].get(i);
				String sglyphChar = Charset.unescapeChar(charIndexer.getObject(ts.getGlyphChar().templateCharIndex));
				transcriptionWithWidthsOutputBuffer.append(sglyphChar + "[" + ts.getGlyphChar().toString(charIndexer) + "][" + w + "]\n");
			}
			transcriptionWithWidthsOutputBuffer.append("\n");
		}
		//System.out.println(transcriptionWithWidthsOutputBuffer.toString());
		FileHelper.writeString(transcriptionWithWidthsOutputFilename, transcriptionWithWidthsOutputBuffer.toString());
		}
		
		//
		//Transcription in ALTO
		//
		{
		System.out.println("Writing alto output to " + altoOutputFilename);
		f.writeString(altoOutputFilename, printAlto(numLines, viterbiTransStates, doc.baseName(), altoOutputFilename, viterbiWidths));
		}

		

		if (text != null) {
			//
			// Evaluate against gold-transcribed data (given as "text")
			//
			@SuppressWarnings("unchecked")
			List<String>[] goldCharSequences = new List[numLines];
			for (int line = 0; line < numLines; ++line) {
				goldCharSequences[line] = new ArrayList<String>();
				for (int i = 0; i < text[line].length; ++i) {
					goldCharSequences[line].add(text[line][i]);
				}
			}

			//
			// Evaluate the comparison
			//
			Map<String, EvalSuffStats> evals = Evaluator.getUnsegmentedEval(viterbiChars, goldCharSequences, charIncludesDiacritic);
			if (allEvals != null) {
				allEvals.add(Tuple2(doc.baseName(), evals));
			}
			
			//
			// Make comparison file
			//
			{
			StringBuffer goldComparisonOutputBuffer = new StringBuffer();
			goldComparisonOutputBuffer.append("MODEL OUTPUT vs. GOLD TRANSCRIPTION\n\n");
			for (int line = 0; line < numLines; ++line) {
				goldComparisonOutputBuffer.append(StringHelper.join(viterbiChars[line], "").trim() + "\n");
				goldComparisonOutputBuffer.append(StringHelper.join(goldCharSequences[line], "").trim() + "\n");
				goldComparisonOutputBuffer.append("\n");
			}
			goldComparisonOutputBuffer.append(Evaluator.renderEval(evals));
			System.out.println("Writing gold comparison to " + goldComparisonOutputFilename);
			//System.out.println(goldComparisonOutputBuffer.toString());
			f.writeString(goldComparisonOutputFilename, goldComparisonOutputBuffer.toString());
			}
			
			//
			// Make comparison file with substitutions
			//
			if (allowGlyphSubstitution) {
			System.out.println("Transcription with substitutions");
			StringBuffer goldComparisonWithSubsOutputBuffer = new StringBuffer();
			goldComparisonWithSubsOutputBuffer.append("MODEL OUTPUT vs. GOLD TRANSCRIPTION\n\n");
			for (int line = 0; line < numLines; ++line) {
				goldComparisonWithSubsOutputBuffer.append(transcriptionWithSubsOutputLines.get(line).trim() + "\n");
				goldComparisonWithSubsOutputBuffer.append(StringHelper.join(goldCharSequences[line], "").trim() + "\n");
				goldComparisonWithSubsOutputBuffer.append("\n");
			}
			goldComparisonWithSubsOutputBuffer.append(Evaluator.renderEval(evals));
			goldComparisonWithSubsOutputBuffer.append("\n\n\n\n\n\n\n");
//			for (TransitionState ts : CollectionHelper.flatten(Arrays.asList(viterbiTransStates))) {
//				goldComparisonWithSubsOutputBuffer.append(ts.toString()).append("\n");
//			}
			System.out.println("Writing gold comparison with substitutions to " + goldComparisonWithSubsOutputFilename);
			System.out.println(goldComparisonWithSubsOutputBuffer.append("\n\n").toString());
			f.writeString(goldComparisonWithSubsOutputFilename, goldComparisonWithSubsOutputBuffer.toString());
			}
		}
		
		if (goldLmChars != null) {
			//
			// Evaluate the comparison
			//
			@SuppressWarnings("unchecked")
			Map<String, EvalSuffStats> lmEvals = Evaluator.getUnsegmentedEval(new List[]{viterbiLmChars}, new List[]{goldLmChars}, charIncludesDiacritic);
			if (allLmEvals != null) {
				allLmEvals.add(Tuple2(doc.baseName(), lmEvals));
			}
			
			//
			// Print LM evaluation
			//
			{
			StringBuffer goldLmComparisonOutputBuffer = new StringBuffer();
			goldLmComparisonOutputBuffer.append("MODEL LM OUTPUT vs. GOLD LM TRANSCRIPTION\n\n");
			goldLmComparisonOutputBuffer.append(StringHelper.join(viterbiLmChars)+"\n");
			goldLmComparisonOutputBuffer.append(StringHelper.join(goldLmChars)+"\n");
			goldLmComparisonOutputBuffer.append(Evaluator.renderEval(lmEvals));
			System.out.println("Writing gold lm comparison to " + goldLmComparisonOutputFilename);
			//System.out.println(goldLmComparisonOutputBuffer.toString());
			f.writeString(goldLmComparisonOutputFilename, goldLmComparisonOutputBuffer.toString());
			}
		}

		if (langIndexer.size() > 1) {
			System.out.println("Multiple languages being used ("+langIndexer.size()+"), so an html file is being generated to show language switching.");
			System.out.println("Writing html output to " + htmlOutputFilename);
			f.writeString(htmlOutputFilename, printLanguageAnnotatedTranscription(numLines, viterbiTransStates, doc.baseName(), htmlOutputFilename));
		}
	}

	private String printLanguageAnnotatedTranscription(int numLines, List<TransitionState>[] viterbiTransStates, String imgFilename, String htmlOutputFilename) {
		StringBuffer outputBuffer = new StringBuffer();
		outputBuffer.append("<HTML xmlns=\"http://www.w3.org/1999/xhtml\" xml:lang=\"en\" lang=\"en\">\n");
		outputBuffer.append("<HEAD><META http-equiv=\"Content-Type\" content=\"text/html; charset=UTF-8\"></HEAD>\n");
		outputBuffer.append("<body>\n");
		outputBuffer.append("<table><tr><td>\n");

		String[] colors = new String[] { "Black", "Red", "Blue", "Olive", "Orange", "Magenta", "Lime", "Cyan", "Purple", "Green", "Brown" };

		int prevLanguage = -1;
		for (int line = 0; line < numLines; ++line) {
			for (TransitionState ts : viterbiTransStates[line]) {
				int lmChar = ts.getLmCharIndex();
				GlyphChar glyph = ts.getGlyphChar();
				int glyphChar = glyph.templateCharIndex;
				String sglyphChar = Charset.unescapeChar(charIndexer.getObject(glyphChar));

				int currLanguage = ts.getLanguageIndex();
				if (currLanguage != prevLanguage) {
					outputBuffer.append("<font color=\"" + colors[currLanguage+1] + "\">");
				}
				
				if (glyph.glyphType == GlyphType.DOUBLED) {
					outputBuffer.append("[2x]");
				}
				else if (lmChar != glyphChar || glyph.glyphType != GlyphType.NORMAL_CHAR)
					outputBuffer.append("[" + Charset.unescapeChar(charIndexer.getObject(lmChar)) + "/" + (glyph.isElided() ? "" : sglyphChar) + "]");
				else
					outputBuffer.append(sglyphChar);
				
				prevLanguage = currLanguage;
			}
			outputBuffer.append("</br>\n");
		}
		outputBuffer.append("</font></font><br/><br/><br/>\n");
		for (int i = -1; i < langIndexer.size(); ++i) {
			outputBuffer.append("<font color=\"" + colors[i+1] + "\">" + (i < 0 ? "none" : langIndexer.getObject(i)) + "</font></br>\n");
		}

		outputBuffer.append("</td><td><img src=\"" + FileUtil.pathRelativeTo(imgFilename, new File(htmlOutputFilename).getParent()) + "\">\n");
		outputBuffer.append("</td></tr></table>\n");
		outputBuffer.append("</body></html>\n");
		outputBuffer.append("\n\n\n");
		outputBuffer.append("\n\n\n\n\n");
		return outputBuffer.toString();
	}


	
//	          <TextLine ID="line_1" WIDTH="619" HEIGHT="105" HPOS="22" VPOS="1560">
//	            <String ID="word_1" WIDTH="65" HEIGHT="70" HPOS="24" VPOS="1560" CONTENT="A" WC="79" emop:DNC="0.0096"></String>
//	          </TextLine>
//	          <TextLine ID="line_2" WIDTH="736" HEIGHT="83" HPOS="132" VPOS="1211">
//	            <String ID="word_3" WIDTH="260" HEIGHT="77" HPOS="132" VPOS="1211" CONTENT="Serious" WC="73" emop:DNC="0.0048"></String>
//	            <SP WIDTH="10"/>
//	            <String ID="word_4" WIDTH="437" HEIGHT="83" HPOS="132" VPOS="1510" CONTENT="Exhortation" WC="72" emop:DNC="0.0048">
//	              <ALTERNATIVE>Exho.rtation</ALTERNATIVE>
//				</String>
//	          </TextLine>
//	          <TextLine ID="line_3" WIDTH="671" HEIGHT="57" HPOS="253" VPOS="1438">
//	            <String ID="word_5" WIDTH="114" HEIGHT="55" HPOS="253" VPOS="1438" CONTENT="To" WC="84" emop:DNC="0.0024">
//	              <ALTERNATIVE>T()</ALTERNATIVE>
//				</String>
//	            <SP WIDTH="10"/>
//	            <String ID="word_6" WIDTH="46" HEIGHT="49" HPOS="260" VPOS="1599" CONTENT="A" WC="89" emop:DNC="0.0048"></String>
//	            <SP WIDTH="10"/>
//	            <String ID="word_7" WIDTH="54" HEIGHT="52" HPOS="258" VPOS="1666" CONTENT="N" WC="92" emop:DNC="0.0083"></String>
//	          </TextLine>
	
	private String printAlto(int numLines, List<TransitionState>[] viterbiTransStates, String imgFilename, String htmlOutputFilename, List<Integer>[] viterbiWidths) {
		StringBuffer outputBuffer = new StringBuffer();
		outputBuffer.append("<alto xmlns=\"http://schema.ccs-gmbh.com/ALTO\" xmlns:emop=\"http://emop.tamu.edu\">\n");
		outputBuffer.append("  <Description>\n");
		outputBuffer.append("    <MeasurementUnit>pixel</MeasurementUnit>\n");
		outputBuffer.append("    <sourceImageInformation>\n");
		outputBuffer.append("      <filename>"+(new File(imgFilename).getName())+"</filename>\n");
		outputBuffer.append("    </sourceImageInformation>\n");
		outputBuffer.append("    <OCRProcessing>\n"); //IDK how we should "ID" this.
		outputBuffer.append("      <preProcessingStep></preProcessingStep>\n"); 
		outputBuffer.append("      <ocrProcessingStep>\n");
		outputBuffer.append("		 <processingDateTime>"+Calendar.getInstance()+"</processingDateTime>\n"); //not really working...
//		for (StringBuffer s: args) {
//			StringBuffer arguments = new StringBuffer();
//			arguments.append(s);
//		};
//		outputBuffer.append("      	 <processingStepSettings>"+arguments+"</processingStepSettings>\n");//want to print "args" here ...
		outputBuffer.append("        <processingSoftware>\n");
		outputBuffer.append("          <softwareCreator>Taylor Berg-Kirkpatrick, Greg Durrett, Dan Klein, Dan Garrette, Hannah Alpert-Abrams</softwareCreator>\n");
		outputBuffer.append("          <softwareName>Ocular</softwareName>\n");
		outputBuffer.append("          <softwareVersion>0.0.3</softwareVersion>\n");
		outputBuffer.append("        </processingSoftware>\n");
		outputBuffer.append("       </ocrProcessingStep>\n");
//		outputBuffer.append("      <postProcessingStep>\n");
//		outputBuffer.append("        <processingSoftware>\n");
//		outputBuffer.append("          <softwareCreator>\n");
//		outputBuffer.append("            Illinois Informatics Institute, University of Illinois at Urbana-Champaign http://www.informatics.illinois.edu\n");
//		outputBuffer.append("          </softwareCreator>\n");
//		outputBuffer.append("          <softwareName>PageCorrector</softwareName>\n");
//		outputBuffer.append("          <softwareVersion>1.10.0-SNAPSHOT</softwareVersion>\n");
//		outputBuffer.append("        </processingSoftware>\n");
//		outputBuffer.append("      </postProcessingStep>\n");
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
		return outputBuffer.toString();
	}

}
