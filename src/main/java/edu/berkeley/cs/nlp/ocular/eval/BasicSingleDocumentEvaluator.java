package edu.berkeley.cs.nlp.ocular.eval;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.HYPHEN;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.LONG_S;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;
import static edu.berkeley.cs.nlp.ocular.util.Tuple3.Tuple3;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import edu.berkeley.cs.nlp.ocular.data.Document;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.eval.Evaluator.EvalSuffStats;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.output.AltoOutputWriter;
import edu.berkeley.cs.nlp.ocular.output.HtmlOutputWriter;
import edu.berkeley.cs.nlp.ocular.sub.GlyphChar;
import edu.berkeley.cs.nlp.ocular.sub.GlyphChar.GlyphType;
import edu.berkeley.cs.nlp.ocular.util.CollectionHelper;
import edu.berkeley.cs.nlp.ocular.util.FileHelper;
import edu.berkeley.cs.nlp.ocular.util.FileUtil;
import edu.berkeley.cs.nlp.ocular.util.StringHelper;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import edu.berkeley.cs.nlp.ocular.util.Tuple3;
import fileio.f;
import indexer.Indexer;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class BasicSingleDocumentEvaluator implements SingleDocumentEvaluator {
	private Indexer<String> charIndexer;
	private Indexer<String> langIndexer;
	private boolean allowGlyphSubstitution;
	private boolean charIncludesDiacritic;
	
	public BasicSingleDocumentEvaluator(Indexer<String> charIndexer, Indexer<String> langIndexer, boolean allowGlyphSubstitution, boolean charIncludesDiacritic) {
		this.charIndexer = charIndexer;
		this.langIndexer = langIndexer;
		this.allowGlyphSubstitution = allowGlyphSubstitution;
		this.charIncludesDiacritic = charIncludesDiacritic;
	}

	private int max(int... xs) {
		int max = Integer.MIN_VALUE;
		for (int x : xs) {
			if (x > max) max = x;
		}
		return max;
	}
	
	private Tuple2<Tuple3<String[][], String[][], List<String>>, Tuple2<TransitionState[][], int[][]>> getGoldTranscriptions(Document doc, TransitionState[][] decodeStates, int[][] decodeWidths) {
		String[][] goldDiplomaticCharLines = doc.loadDiplomaticTextLines();
		String[][] goldNormalizedCharLines = doc.loadNormalizedTextLines();
		List<String> goldNormalizedChars = doc.loadNormalizedText();
		
		//
		// Make sure the decoded states and the text have the same number of lines (numLines)
		//
		int numLines = max(decodeStates.length, goldDiplomaticCharLines.length, goldNormalizedCharLines.length);
		if (goldDiplomaticCharLines.length < numLines) { // if we need to pad the end with blank lines
			String[][] newText = new String[numLines][];
			for (int line = 0; line < numLines; ++line) {
				newText[line] = line < goldDiplomaticCharLines.length ? goldDiplomaticCharLines[line] : new String[0];
			}
			goldDiplomaticCharLines = newText;
		}
		if (goldNormalizedCharLines.length < numLines) { // if we need to pad the end with blank lines
			String[][] newText = new String[numLines][];
			for (int line = 0; line < numLines; ++line) {
				newText[line] = line < goldNormalizedCharLines.length ? goldNormalizedCharLines[line] : new String[0];
			}
			goldNormalizedCharLines = newText;
		}
		if (decodeStates.length < numLines) { // if we need to pad the end with blank lines
			numLines = goldDiplomaticCharLines.length;
			TransitionState[][] newDecodeStates = new TransitionState[numLines][];
			int[][] newDeocdeWidths = new int[numLines][];
			for (int line = 0; line < numLines; ++line) {
				newDecodeStates[line] = line < decodeStates.length ? decodeStates[line] : new TransitionState[0];
				newDeocdeWidths[line] = line < decodeWidths.length ? decodeWidths[line] : new int[0];
			}
			decodeStates = newDecodeStates;
			decodeWidths = newDeocdeWidths;
		}
		
		return Tuple2(Tuple3(goldDiplomaticCharLines, goldNormalizedCharLines, goldNormalizedChars), Tuple2(decodeStates, decodeWidths));
	}
	
	private <A> List<A>[] toArrayOfLists(A[][] as) {
		@SuppressWarnings("unchecked")
		List<A>[] r = new List[as.length];
		for (int i = 0; i < as.length; ++i) { 
			r[i] = Arrays.asList(as[i]);
		}
		return r;
	}
	
	public void printTranscriptionWithEvaluation(int iter, int batchId,
			Document doc,
			TransitionState[][] decodeStates, int[][] decodeWidths,
			String inputDocPath, String outputPath,
			List<Tuple2<String, Map<String, EvalSuffStats>>> allDiplomaticEvals,
			List<Tuple2<String, Map<String, EvalSuffStats>>> allNormalizedEvals) {
		
		Tuple2<Tuple3<String[][], String[][], List<String>>, Tuple2<TransitionState[][], int[][]>> goldTranscriptionData = getGoldTranscriptions(doc, decodeStates, decodeWidths);
		String[][] goldDiplomaticLineChars = goldTranscriptionData._1._1;
		String[][] goldNormalizedLineChars = goldTranscriptionData._1._2;
		List<String> goldNormalizedChars = goldTranscriptionData._1._3;
		decodeStates = goldTranscriptionData._2._1;
		decodeWidths = goldTranscriptionData._2._2;
		
		int numLines = decodeStates.length;
		
		//
		// Get the model output
		//
		@SuppressWarnings("unchecked")
		List<String>[] viterbiDiplomaticCharLines = new List[numLines];
		@SuppressWarnings("unchecked")
		List<String>[] viterbiNormalizedCharLines = new List[numLines];
		List<String> viterbiNormalizedTranscription = new ArrayList<String>(); // A continuous string, re-assembling words hyphenated over a line.
		@SuppressWarnings("unchecked")
		List<TransitionState>[] viterbiTransStates = new List[numLines];
		@SuppressWarnings("unchecked")
		List<Integer>[] viterbiWidths = new List[numLines];
		
		for (int line = 0; line < numLines; ++line) {
			
			viterbiDiplomaticCharLines[line] = new ArrayList<String>();
			viterbiNormalizedCharLines[line] = new ArrayList<String>();
			viterbiTransStates[line] = new ArrayList<TransitionState>();
			viterbiWidths[line] = new ArrayList<Integer>();
			
			for (int i = 0; i < decodeStates[line].length; ++i) {
				TransitionState ts = decodeStates[line][i];
				String currDiplomaticChar = charIndexer.getObject(ts.getGlyphChar().templateCharIndex);
				String prevDiplomaticChar = CollectionHelper.last(viterbiDiplomaticCharLines[line]);
				if (HYPHEN.equals(prevDiplomaticChar) && HYPHEN.equals(currDiplomaticChar)) {
					// collapse multi-hyphens
				}
				else {

					//
					// Add diplomatic characters to diplomatic transcription
					//
					if (!ts.getGlyphChar().isElided()) {
						viterbiDiplomaticCharLines[line].add(currDiplomaticChar);
					}
					
					//
					// Add normalized characters to normalized transcriptions
					//
					if (ts.getGlyphChar().glyphType != GlyphType.DOUBLED) { // the first in a pair of doubled characters isn't part of the normalized transcription
						String currNormalizedChar = charIndexer.getObject(ts.getLmCharIndex());
						if (LONG_S.equals(currNormalizedChar)) currNormalizedChar = "s"; // don't use long-s in normalized transcriptions
						
						//
						// Add to normalized line transcription
						viterbiNormalizedCharLines[line].add(currNormalizedChar);
						
						//
						// Add to normalized running transcription
						switch(ts.getType()) {
							case RMRGN_HPHN_INIT:
							case RMRGN_HPHN:
							case LMRGN_HPHN:
								break;
								
							case LMRGN:
							case RMRGN:
								viterbiNormalizedTranscription.add(" ");
								break;
							
							case TMPL:
								viterbiNormalizedTranscription.add(currNormalizedChar);
						}
					}
				}
					
				viterbiTransStates[line].add(ts);
				viterbiWidths[line].add(decodeWidths[line][i]);
			}
		}
		
		String outputFilenameBase = makeOutputFilenameBase(iter, batchId, doc, inputDocPath, outputPath);
		new File(outputFilenameBase).getParentFile().mkdirs();
		
		//
		// Evaluate the comparison
		//
		Map<String, EvalSuffStats> diplomaticEvals = null;
		if (goldDiplomaticLineChars != null) {
			Evaluator.getUnsegmentedEval(viterbiDiplomaticCharLines, toArrayOfLists(goldDiplomaticLineChars), charIncludesDiacritic);
			if (allDiplomaticEvals != null) allDiplomaticEvals.add(Tuple2(doc.baseName(), diplomaticEvals));
		}

		Map<String, EvalSuffStats> normalizedEvals = null;
		if (goldNormalizedLineChars != null) {
			Evaluator.getUnsegmentedEval(viterbiNormalizedCharLines, toArrayOfLists(goldNormalizedLineChars), charIncludesDiacritic);
			if (allNormalizedEvals != null) allNormalizedEvals.add(Tuple2(doc.baseName(), normalizedEvals));
		}
		
		
		//
		// Diplomatic transcription output
		//
		{
			String transcriptionOutputFilename = outputFilenameBase + "_transcription.txt";
			
			StringBuffer transcriptionOutputBuffer = new StringBuffer();
			for (int line = 0; line < numLines; ++line) {
				transcriptionOutputBuffer.append(StringHelper.join(viterbiDiplomaticCharLines[line], "") + "\n");
			}
			
			System.out.println(transcriptionOutputBuffer.toString() + "\n\n");
			
			System.out.println("Writing transcription output to " + transcriptionOutputFilename);
			FileHelper.writeString(transcriptionOutputFilename, transcriptionOutputBuffer.toString());
		}

		//
		// Normalized transcription lines output
		//
		if (allowGlyphSubstitution) {
			String transcriptionOutputFilename = outputFilenameBase + "_transcription_normalized_lines.txt";
			
			StringBuffer transcriptionOutputBuffer = new StringBuffer();
			for (int line = 0; line < numLines; ++line) {
				transcriptionOutputBuffer.append(StringHelper.join(viterbiNormalizedCharLines[line], "") + "\n");
			}
			
			System.out.println(transcriptionOutputBuffer.toString() + "\n\n");
			
			System.out.println("Writing normalized transcription lines output to " + transcriptionOutputFilename);
			FileHelper.writeString(transcriptionOutputFilename, transcriptionOutputBuffer.toString());
		}
		
		//
		// Normalized transcription cleaned output
		//
		if (allowGlyphSubstitution) {
			String transcriptionOutputFilename = outputFilenameBase + "_transcription_normalized.txt";
			
			String transcriptionOutputBuffer = StringHelper.join(viterbiNormalizedTranscription);
			
			System.out.println(transcriptionOutputBuffer.toString() + "\n\n");
			
			System.out.println("Writing normalized transcription output to " + transcriptionOutputFilename);
			FileHelper.writeString(transcriptionOutputFilename, transcriptionOutputBuffer.toString());
		}
		
		//
		// Make comparison file
		//
		if (allowGlyphSubstitution || goldDiplomaticLineChars != null || goldNormalizedChars != null) {
			String transcriptionOutputFilename = outputFilenameBase + "_comparisons.txt";
			
			List<String> transcriptionWithSubsOutputLines = getTranscriptionLinesWithSubs(viterbiTransStates);
			
			StringBuffer goldComparisonOutputBuffer = new StringBuffer();
			if (allowGlyphSubstitution) goldComparisonOutputBuffer.append("Model transcription with substitutions\n");
			/*                       */ goldComparisonOutputBuffer.append("Model diplomatic transcription\n");
			if (allowGlyphSubstitution) goldComparisonOutputBuffer.append("Model normalized transcription\n");
			if (goldDiplomaticLineChars != null) goldComparisonOutputBuffer.append("Gold diplomatic transcription\n");
			if (goldNormalizedChars != null) goldComparisonOutputBuffer.append("Gold normalized transcription\n");
			
			for (int line = 0; line < numLines; ++line) {
				if (allowGlyphSubstitution) goldComparisonOutputBuffer.append(transcriptionWithSubsOutputLines+"\n");
				/*                       */ goldComparisonOutputBuffer.append(StringHelper.join(viterbiDiplomaticCharLines[line]).trim() + "\n");
				if (allowGlyphSubstitution) goldComparisonOutputBuffer.append(StringHelper.join(viterbiNormalizedCharLines[line]).trim() + "\n");
				if (goldDiplomaticLineChars != null) goldComparisonOutputBuffer.append(StringHelper.join(goldDiplomaticLineChars[line]).trim() + "\n");
				if (goldNormalizedLineChars != null && allowGlyphSubstitution) goldComparisonOutputBuffer.append(StringHelper.join(goldNormalizedLineChars[line]).trim() + "\n");
				goldComparisonOutputBuffer.append("\n");
			}
			if (goldNormalizedChars != null && allowGlyphSubstitution) goldComparisonOutputBuffer.append("\n" + StringHelper.join(goldNormalizedChars).trim() + "\n\n");
			
			if (goldDiplomaticLineChars != null) {
				goldComparisonOutputBuffer.append("Diplomatic evaluation\n");
				goldComparisonOutputBuffer.append(Evaluator.renderEval(diplomaticEvals));
			}
			if (goldDiplomaticLineChars != null) {
				goldComparisonOutputBuffer.append("Normalized evaluation\n");
				goldComparisonOutputBuffer.append(Evaluator.renderEval(diplomaticEvals));
			}
			
			System.out.println("Writing comparisons to " + transcriptionOutputFilename);
			//System.out.println(goldComparisonOutputBuffer.toString());
			f.writeString(transcriptionOutputFilename, goldComparisonOutputBuffer.toString());
		}
		
		//
		// Other files
		//
		new AltoOutputWriter(charIndexer, langIndexer).write(numLines, viterbiTransStates, inputDocPath, outputFilenameBase, viterbiWidths);
		new HtmlOutputWriter(charIndexer, langIndexer).write(numLines, viterbiTransStates, inputDocPath, outputFilenameBase);
	
		//
		// Transcription with widths
		//
//		if (allowGlyphSubstitution) {
//		System.out.println("Transcription with widths");
//		StringBuffer transcriptionWithWidthsOutputBuffer = new StringBuffer();
//		for (int line = 0; line < numLines; ++line) {
//			transcriptionWithWidthsOutputBuffer.append(transcriptionWithSubsOutputLines.get(line));
//			for (int i = 0; i < viterbiTransStates[line].size(); ++i) {
//				TransitionState ts = viterbiTransStates[line].get(i);
//				int w = viterbiWidths[line].get(i);
//				String sglyphChar = Charset.unescapeChar(charIndexer.getObject(ts.getGlyphChar().templateCharIndex));
//				transcriptionWithWidthsOutputBuffer.append(sglyphChar + "[" + ts.getGlyphChar().toString(charIndexer) + "][" + w + "]\n");
//			}
//			transcriptionWithWidthsOutputBuffer.append("\n");
//		}
//		//System.out.println(transcriptionWithWidthsOutputBuffer.toString());
//		FileHelper.writeString(transcriptionWithWidthsOutputFilename, transcriptionWithWidthsOutputBuffer.toString());
//		}
	}

	private String makeOutputFilenameBase(int iter, int batchId, Document doc, String inputDocPath, String outputPath) {
		String fileParent = FileUtil.removeCommonPathPrefixOfParents(new File(inputDocPath), new File(doc.baseName()))._2;
		String preext = FileUtil.withoutExtension(new File(doc.baseName()).getName());
		String outputFilenameBase = outputPath + "/" + fileParent + "/" + preext;
		if (iter > 0) outputFilenameBase += "_iter-" + iter;
		if (batchId > 0) outputFilenameBase += "_batch-" + batchId;
		return outputFilenameBase;
	}

	private List<String> getTranscriptionLinesWithSubs(List<TransitionState>[] viterbiTransStates) {
		List<String> transcriptionWithSubsOutputLines = new ArrayList<String>();
		for (List<TransitionState> lineStates : viterbiTransStates) {
			StringBuilder lineBuffer = new StringBuilder();
			for (TransitionState ts : lineStates) {
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
		return transcriptionWithSubsOutputLines;
	}

}
