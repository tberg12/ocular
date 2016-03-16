package edu.berkeley.cs.nlp.ocular.eval;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.HYPHEN;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.LONG_S;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.SPACE;
import static edu.berkeley.cs.nlp.ocular.util.CollectionHelper.last;
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
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar.GlyphType;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.output.AltoOutputWriter;
import edu.berkeley.cs.nlp.ocular.output.HtmlOutputWriter;
import edu.berkeley.cs.nlp.ocular.util.ArrayHelper;
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
public class BasicSingleDocumentEvaluatorAndOutputPrinter implements SingleDocumentEvaluatorAndOutputPrinter {
	private Indexer<String> charIndexer;
	private Indexer<String> langIndexer;
	private boolean allowGlyphSubstitution;
	private boolean charIncludesDiacritic;
	
	public BasicSingleDocumentEvaluatorAndOutputPrinter(Indexer<String> charIndexer, Indexer<String> langIndexer, boolean allowGlyphSubstitution, boolean charIncludesDiacritic) {
		this.charIndexer = charIndexer;
		this.langIndexer = langIndexer;
		this.allowGlyphSubstitution = allowGlyphSubstitution;
		this.charIncludesDiacritic = charIncludesDiacritic;
	}

	public Tuple2<Map<String, EvalSuffStats>,Map<String, EvalSuffStats>> evaluateAndPrintTranscription(int iter, int batchId,
			Document doc,
			TransitionState[][] decodeStates, int[][] decodeWidths,
			String inputDocPath, String outputPath) {
		
		Tuple2<Tuple3<String[][], String[][], List<String>>, Tuple2<TransitionState[][], int[][]>> goldTranscriptionData = loadGoldTranscriptions(doc, decodeStates, decodeWidths);
		String[][] goldDiplomaticLineChars = goldTranscriptionData._1._1;
		String[][] goldNormalizedLineChars = goldTranscriptionData._1._2;
		List<String> goldNormalizedChars = goldTranscriptionData._1._3;
		decodeStates = goldTranscriptionData._2._1;
		decodeWidths = goldTranscriptionData._2._2;
		
		int numLines = decodeStates.length;
		
		//
		// Get the model output
		//
		ModelTranscriptions mt = new ModelTranscriptions(numLines, decodeStates, decodeWidths, charIndexer);
		
		
		String outputFilenameBase = makeOutputFilenameBase(iter, batchId, doc, inputDocPath, outputPath);
		new File(outputFilenameBase).getParentFile().mkdirs();
		
		//
		// Evaluate the comparison
		//
		Map<String, EvalSuffStats> diplomaticEvals = goldDiplomaticLineChars != null ? Evaluator.getUnsegmentedEval(mt.viterbiDiplomaticCharLines, toArrayOfLists(goldDiplomaticLineChars), charIncludesDiacritic) : null;
		Map<String, EvalSuffStats> normalizedEvals = goldNormalizedLineChars != null ? Evaluator.getUnsegmentedEval(mt.viterbiNormalizedCharLines, toArrayOfLists(goldNormalizedLineChars), charIncludesDiacritic) : null;
		
		
		//
		// Diplomatic transcription output
		//
		{
			String transcriptionOutputFilename = diplomaticTranscriptionOutputFile(outputFilenameBase);
			
			StringBuffer transcriptionOutputBuffer = new StringBuffer();
			for (int line = 0; line < numLines; ++line) {
				transcriptionOutputBuffer.append(StringHelper.join(mt.viterbiDiplomaticCharLines[line]) + "\n");
			}
			
			System.out.println("\n" + transcriptionOutputBuffer.toString());
			
			System.out.println("Writing transcription output to " + transcriptionOutputFilename);
			FileHelper.writeString(transcriptionOutputFilename, transcriptionOutputBuffer.toString());
		}

		//
		// Normalized transcription lines output
		//
		if (allowGlyphSubstitution) {
			String transcriptionOutputFilename = normalizedLinesTranscriptionOutputFile(outputFilenameBase);
			
			StringBuffer transcriptionOutputBuffer = new StringBuffer();
			for (int line = 0; line < numLines; ++line) {
				transcriptionOutputBuffer.append(StringHelper.join(mt.viterbiNormalizedCharLines[line]) + "\n");
			}
			
			//System.out.println("\n" + transcriptionOutputBuffer.toString());
			
			System.out.println("Writing normalized transcription lines output to " + transcriptionOutputFilename);
			FileHelper.writeString(transcriptionOutputFilename, transcriptionOutputBuffer.toString());
		}
		
		//
		// Normalized transcription cleaned output
		//
		if (allowGlyphSubstitution) {
			String transcriptionOutputFilename = normalizedTranscriptionOutputFile(outputFilenameBase);
			
			String transcriptionOutputBuffer = StringHelper.join(mt.viterbiNormalizedTranscription);
			
			//System.out.println("\n" + transcriptionOutputBuffer.toString() + "\n");
			
			System.out.println("Writing normalized transcription output to " + transcriptionOutputFilename);
			FileHelper.writeString(transcriptionOutputFilename, transcriptionOutputBuffer.toString());
		}
		
		//
		// Make comparison file
		//
		if (allowGlyphSubstitution || goldDiplomaticLineChars != null || goldNormalizedLineChars != null) {
			String transcriptionOutputFilename = comparisonsTranscriptionOutputFile(outputFilenameBase);
			
			List<String> transcriptionWithSubsOutputLines = getTranscriptionLinesWithSubs(mt.viterbiTransStates);
			
			StringBuffer goldComparisonOutputBuffer = new StringBuffer();
			if (allowGlyphSubstitution)          goldComparisonOutputBuffer.append("MN: " + "Model normalized transcription\n");
			if (goldNormalizedLineChars != null) goldComparisonOutputBuffer.append("GN: " + "Gold normalized transcription\n");
			/*                       */          goldComparisonOutputBuffer.append("MD: " + "Model diplomatic transcription\n");
			if (goldDiplomaticLineChars != null) goldComparisonOutputBuffer.append("GD: " + "Gold diplomatic transcription\n");
			if (allowGlyphSubstitution)          goldComparisonOutputBuffer.append("MS: " + "Model transcription with substitutions\n");
			goldComparisonOutputBuffer.append("\n\n");
			
			for (int line = 0; line < numLines; ++line) {
				if (allowGlyphSubstitution)          goldComparisonOutputBuffer.append("MN: " + StringHelper.join(mt.viterbiNormalizedCharLines[line]).trim() + "\n");
				if (goldNormalizedLineChars != null) goldComparisonOutputBuffer.append("GN: " + StringHelper.join(goldNormalizedLineChars[line]).trim() + "\n");
				/*                       */          goldComparisonOutputBuffer.append("MD: " + StringHelper.join(mt.viterbiDiplomaticCharLines[line]).trim() + "\n");
				if (goldDiplomaticLineChars != null) goldComparisonOutputBuffer.append("GD: " + StringHelper.join(goldDiplomaticLineChars[line]).trim() + "\n");
				if (allowGlyphSubstitution)          goldComparisonOutputBuffer.append("MS: " + transcriptionWithSubsOutputLines.get(line).trim()+"\n");
				goldComparisonOutputBuffer.append("\n");
			}
			goldComparisonOutputBuffer.append("\n");
			
			if (mt.viterbiNormalizedTranscription != null || goldNormalizedChars != null) {
				if (mt.viterbiNormalizedTranscription != null && goldNormalizedChars != null) {
					goldComparisonOutputBuffer.append("Model (top) vs. Gold (bottom) normalized transcriptions\n");
				}
				else if (mt.viterbiNormalizedTranscription != null) {
					goldComparisonOutputBuffer.append("Model normalized transcription\n");
				}
				else if (goldNormalizedChars != null) {
					goldComparisonOutputBuffer.append("Gold normalized transcription\n");
				}
				
				if (mt.viterbiNormalizedTranscription != null) {
					goldComparisonOutputBuffer.append(StringHelper.join(mt.viterbiNormalizedTranscription) + "\n");
				}
				if (goldNormalizedChars != null) {
					goldComparisonOutputBuffer.append(StringHelper.join(goldNormalizedChars) + "\n");
				}
			}
			
			goldComparisonOutputBuffer.append("\n");
			if (goldDiplomaticLineChars != null) {
				goldComparisonOutputBuffer.append("\nDiplomatic evaluation\n");
				goldComparisonOutputBuffer.append(Evaluator.renderEval(diplomaticEvals));
			}
			if (goldNormalizedLineChars != null) {
				goldComparisonOutputBuffer.append("\nNormalized evaluation\n");
				goldComparisonOutputBuffer.append(Evaluator.renderEval(normalizedEvals));
			}
			
			System.out.println("Writing comparisons to " + transcriptionOutputFilename);
			//System.out.println(goldComparisonOutputBuffer.toString());
			f.writeString(transcriptionOutputFilename, goldComparisonOutputBuffer.toString());
		}
		
		//
		// Other files
		//
		new AltoOutputWriter(charIndexer, langIndexer).write(numLines, mt.viterbiTransStates, inputDocPath, outputFilenameBase, mt.viterbiWidths);
		new HtmlOutputWriter(charIndexer, langIndexer).write(numLines, mt.viterbiTransStates, inputDocPath, outputFilenameBase);
	
		//
		// Transcription with widths
		//
//		if (allowGlyphSubstitution) {
//		System.out.println("Transcription with widths");
//		StringBuffer transcriptionWithWidthsOutputBuffer = new StringBuffer();
//		for (int line = 0; line < numLines; ++line) {
//			transcriptionWithWidthsOutputBuffer.append(transcriptionWithSubsOutputLines.get(line));
//			for (int i = 0; i < viterbiTrmt.viterbies[line].size(); ++i) {
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
		
		return Tuple2(diplomaticEvals, normalizedEvals);
	}

	private Tuple2<Tuple3<String[][], String[][], List<String>>, Tuple2<TransitionState[][], int[][]>> loadGoldTranscriptions(Document doc, TransitionState[][] decodeStates, int[][] decodeWidths) {
		String[][] goldDiplomaticCharLines = doc.loadDiplomaticTextLines();
		String[][] goldNormalizedCharLines = doc.loadNormalizedTextLines();
		List<String> goldNormalizedChars = doc.loadNormalizedText();
		
		//
		// Make sure the decoded states and the text have the same number of lines (numLines)
		//
		int numLines = ArrayHelper.max(
				decodeStates.length, 
				goldDiplomaticCharLines != null ? goldDiplomaticCharLines.length : 0, 
				goldNormalizedCharLines != null ? goldNormalizedCharLines.length : 0);
		
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
		if (goldDiplomaticCharLines != null && goldDiplomaticCharLines.length < numLines) { // if we need to pad the end with blank lines
			String[][] newText = new String[numLines][];
			for (int line = 0; line < numLines; ++line) {
				newText[line] = line < goldDiplomaticCharLines.length ? goldDiplomaticCharLines[line] : new String[0];
			}
			goldDiplomaticCharLines = newText;
		}
		if (goldNormalizedCharLines != null && goldNormalizedCharLines.length < numLines) { // if we need to pad the end with blank lines
			String[][] newText = new String[numLines][];
			for (int line = 0; line < numLines; ++line) {
				newText[line] = line < goldNormalizedCharLines.length ? goldNormalizedCharLines[line] : new String[0];
			}
			goldNormalizedCharLines = newText;
		}
		
		return Tuple2(Tuple3(goldDiplomaticCharLines, goldNormalizedCharLines, goldNormalizedChars), Tuple2(decodeStates, decodeWidths));
	}
	
	public static String makeOutputFilenameBase(Document doc, String inputDocPath, String outputPath) {
		return makeOutputFilenameBase(0, 0, doc, inputDocPath, outputPath);
	}
	private static String makeOutputFilenameBase(int iter, int batchId, Document doc, String inputDocPath, String outputPath) {
		String fileParent = FileUtil.removeCommonPathPrefixOfParents(new File(inputDocPath), new File(doc.baseName()))._2;
		String preext = FileUtil.withoutExtension(new File(doc.baseName()).getName());
		String outputFilenameBase = outputPath + "/all_transcriptions/" + fileParent + "/" + preext;
		if (iter > 0) outputFilenameBase += "_iter-" + iter;
		if (batchId > 0) outputFilenameBase += "_batch-" + batchId;
		return outputFilenameBase;
	}
	
	public static String diplomaticTranscriptionOutputFile /*     */ (String outputFilenameBase) { return outputFilenameBase + "_transcription.txt"; }
	public static String normalizedLinesTranscriptionOutputFile /**/ (String outputFilenameBase) { return outputFilenameBase + "_transcription_normalized_lines.txt"; }
	public static String normalizedTranscriptionOutputFile /*     */ (String outputFilenameBase) { return outputFilenameBase + "_transcription_normalized.txt"; }
	public static String comparisonsTranscriptionOutputFile /*    */ (String outputFilenameBase) { return outputFilenameBase + "_comparisons.txt"; }


	private static class ModelTranscriptions {
		public final List<String>[] viterbiDiplomaticCharLines;
		public final List<String>[] viterbiNormalizedCharLines;
		public final List<String> viterbiNormalizedTranscription; // A continuous string, re-assembling words hyphenated over a line.
		public final List<TransitionState>[] viterbiTransStates;
		public final List<Integer>[] viterbiWidths;

		@SuppressWarnings("unchecked")
		public ModelTranscriptions(int numLines, TransitionState[][] decodeStates, int[][] decodeWidths, Indexer<String> charIndexer) {
			this.viterbiDiplomaticCharLines = new List[numLines];
			this.viterbiNormalizedCharLines = new List[numLines];
			this.viterbiNormalizedTranscription = new ArrayList<String>();
			this.viterbiTransStates = new List[numLines];
			this.viterbiWidths = new List[numLines];

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
							viterbiDiplomaticCharLines[line].add(Charset.unescapeCharPrecomposedOnly(currDiplomaticChar));
						}
						
						//
						// Add normalized characters to normalized transcriptions
						//
						if (ts.getGlyphChar().glyphType != GlyphType.DOUBLED) { // the first in a pair of doubled characters isn't part of the normalized transcription
							String currNormalizedChar = charIndexer.getObject(ts.getLmCharIndex());
							if (LONG_S.equals(currNormalizedChar)) currNormalizedChar = "s"; // don't use long-s in normalized transcriptions
							
							//
							// Add to normalized line transcription
							viterbiNormalizedCharLines[line].add(Charset.unescapeCharPrecomposedOnly(currNormalizedChar));
							
							//
							// Add to normalized running transcription
							switch(ts.getType()) {
								case RMRGN_HPHN_INIT:
								case RMRGN_HPHN:
								case LMRGN_HPHN:
									break;
									
								case LMRGN:
								case RMRGN:
									if (!viterbiNormalizedTranscription.isEmpty() && !SPACE.equals(last(viterbiNormalizedTranscription)))
										viterbiNormalizedTranscription.add(SPACE);
									break;
								
								case TMPL:
									if (SPACE.equals(currNormalizedChar) && (viterbiNormalizedTranscription.isEmpty() || SPACE.equals(last(viterbiNormalizedTranscription)))) {
										// do nothing -- collapse spaces
									}
									else {
										viterbiNormalizedTranscription.add(Charset.unescapeCharPrecomposedOnly(currNormalizedChar));
									}
							}
						}
					}
						
					viterbiTransStates[line].add(ts);
					viterbiWidths[line].add(decodeWidths[line][i]);
				}
			}

			if (SPACE.equals(last(viterbiNormalizedTranscription))) {
				viterbiNormalizedTranscription.remove(viterbiNormalizedTranscription.size()-1);
			}
		}
	}
	
	private List<String> getTranscriptionLinesWithSubs(List<TransitionState>[] viterbiTransStates) {
		List<String> transcriptionWithSubsOutputLines = new ArrayList<String>();
		for (List<TransitionState> lineStates : viterbiTransStates) {
			StringBuilder lineBuffer = new StringBuilder();
			for (TransitionState ts : lineStates) {
				int lmChar = ts.getLmCharIndex();
				GlyphChar glyph = ts.getGlyphChar();
				int glyphChar = glyph.templateCharIndex;
				String sglyphChar = Charset.unescapeCharPrecomposedOnly(charIndexer.getObject(glyphChar));
				if (lmChar != glyphChar || glyph.glyphType != GlyphType.NORMAL_CHAR) {
					String norm = Charset.unescapeCharPrecomposedOnly(charIndexer.getObject(lmChar));
					String dipl = (glyph.glyphType == GlyphType.DOUBLED ? "2x"+sglyphChar : glyph.isElided() ? "" : sglyphChar);
					lineBuffer.append("[" + norm + "/" + dipl + "]");
				}
				else {
					lineBuffer.append(sglyphChar);
				}
			}
			transcriptionWithSubsOutputLines.add(lineBuffer.toString());
		}
		return transcriptionWithSubsOutputLines;
	}

	private <A> List<A>[] toArrayOfLists(A[][] as) {
		@SuppressWarnings("unchecked")
		List<A>[] r = new List[as.length];
		for (int i = 0; i < as.length; ++i) { 
			r[i] = Arrays.asList(as[i]);
		}
		return r;
	}

}
