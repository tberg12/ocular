package edu.berkeley.cs.nlp.ocular.eval;

import static edu.berkeley.cs.nlp.ocular.main.FonttrainTranscribeShared.OutputFormat.ALTO;
import static edu.berkeley.cs.nlp.ocular.main.FonttrainTranscribeShared.OutputFormat.COMP;
import static edu.berkeley.cs.nlp.ocular.main.FonttrainTranscribeShared.OutputFormat.DIPL;
import static edu.berkeley.cs.nlp.ocular.main.FonttrainTranscribeShared.OutputFormat.HTML;
import static edu.berkeley.cs.nlp.ocular.main.FonttrainTranscribeShared.OutputFormat.NORM;
import static edu.berkeley.cs.nlp.ocular.main.FonttrainTranscribeShared.OutputFormat.NORMLINES;
import static edu.berkeley.cs.nlp.ocular.main.FonttrainTranscribeShared.OutputFormat.WHITESPACE;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;
import static edu.berkeley.cs.nlp.ocular.util.Tuple3.Tuple3;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.data.Document;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.eval.Evaluator.EvalSuffStats;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphChar.GlyphType;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.main.FonttrainTranscribeShared.OutputFormat;
import edu.berkeley.cs.nlp.ocular.model.DecodeState;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.output.AltoOutputWriter;
import edu.berkeley.cs.nlp.ocular.output.HtmlOutputWriter;
import edu.berkeley.cs.nlp.ocular.util.ArrayHelper;
import edu.berkeley.cs.nlp.ocular.util.FileHelper;
import edu.berkeley.cs.nlp.ocular.util.FileUtil;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import edu.berkeley.cs.nlp.ocular.util.Tuple3;
import tberg.murphy.fileio.f;
import tberg.murphy.indexer.Indexer;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class BasicSingleDocumentEvaluatorAndOutputPrinter implements SingleDocumentEvaluatorAndOutputPrinter {
	private Indexer<String> charIndexer;
	private Indexer<String> langIndexer;
	private boolean allowGlyphSubstitution;
	private boolean charIncludesDiacritic;
	private List<String> commandLineArgs;
	
	public BasicSingleDocumentEvaluatorAndOutputPrinter(Indexer<String> charIndexer, Indexer<String> langIndexer, boolean allowGlyphSubstitution, boolean charIncludesDiacritic, List<String> commandLineArgs) {
		this.charIndexer = charIndexer;
		this.langIndexer = langIndexer;
		this.allowGlyphSubstitution = allowGlyphSubstitution;
		this.charIncludesDiacritic = charIncludesDiacritic;
		this.commandLineArgs = commandLineArgs;
	}

	private String joinLineForPrinting(List<String> chars) {
		StringBuilder b = new StringBuilder();
		for (String c : chars)
			b.append(Charset.unescapeChar(c));
		return b.toString();
	}
	
	public Tuple2<Map<String, EvalSuffStats>,Map<String, EvalSuffStats>> evaluateAndPrintTranscription(int iter, int batchId,
			Document doc,
			DecodeState[][] decodeStates,
			String inputDocPath, String outputPath, Set<OutputFormat> outputFormats,
			CodeSwitchLanguageModel lm) {
		
		Tuple2<Tuple3<String[][], String[][], List<String>>, DecodeState[][]> goldTranscriptionData = loadGoldTranscriptions(doc, decodeStates);
		String[][] goldDiplomaticLineChars = goldTranscriptionData._1._1;
		String[][] goldNormalizedLineChars = goldTranscriptionData._1._2;
		List<String> goldNormalizedChars = goldTranscriptionData._1._3;
		decodeStates = goldTranscriptionData._2; // in case we needed to add blank rows
		
		int numLines = decodeStates.length;
		
		//
		// Get the model output
		//
		ModelTranscriptions mt = new ModelTranscriptions(decodeStates, charIndexer, langIndexer);
		
		
		String outputFilenameBase = makeOutputFilenameBase(iter, batchId, doc, inputDocPath, outputPath);
		new File(outputFilenameBase).getAbsoluteFile().getParentFile().mkdirs();
		
		//
		// Evaluate the comparison
		//
		Map<String, EvalSuffStats> diplomaticEvals = goldDiplomaticLineChars != null ? Evaluator.getUnsegmentedEval(mt.getViterbiDiplomaticCharLines(), toArrayOfLists(goldDiplomaticLineChars), charIncludesDiacritic) : null;
		Map<String, EvalSuffStats> normalizedEvals = goldNormalizedLineChars != null ? Evaluator.getUnsegmentedEval(mt.getViterbiNormalizedCharLines(), toArrayOfLists(goldNormalizedLineChars), charIncludesDiacritic) : null;
		
		
		//
		// Diplomatic transcription output
		//
		{
			String transcriptionOutputFilename = diplomaticTranscriptionOutputFile(outputFilenameBase);
			
			StringBuffer transcriptionOutputBuffer = new StringBuffer();
			for (int line = 0; line < numLines; ++line) {
				transcriptionOutputBuffer.append(joinLineForPrinting(mt.getViterbiDiplomaticCharLines()[line]) + "\n");
			}
			
			System.out.println("\n" + transcriptionOutputBuffer.toString());
			
			if (outputFormats.contains(DIPL)) {	
				System.out.println("Writing transcription output to " + transcriptionOutputFilename);
				FileHelper.writeString(transcriptionOutputFilename, transcriptionOutputBuffer.toString());
			}
		}

		//
		// Normalized transcription lines output
		//
		if (allowGlyphSubstitution) {
			String transcriptionOutputFilename = normalizedLinesTranscriptionOutputFile(outputFilenameBase);
			
			StringBuffer transcriptionOutputBuffer = new StringBuffer();
			for (int line = 0; line < numLines; ++line) {
				transcriptionOutputBuffer.append(joinLineForPrinting(mt.getViterbiNormalizedCharLines()[line]) + "\n");
			}
			
			//System.out.println("\n" + transcriptionOutputBuffer.toString());
			
			if (outputFormats.contains(NORMLINES)) {
				System.out.println("Writing normalized transcription lines output to " + transcriptionOutputFilename);
				FileHelper.writeString(transcriptionOutputFilename, transcriptionOutputBuffer.toString());
			}
		}
		
		//
		// Normalized transcription cleaned output
		//
		if (allowGlyphSubstitution) {
			String transcriptionOutputFilename = normalizedTranscriptionOutputFile(outputFilenameBase);
			
			String transcriptionOutputBuffer = joinLineForPrinting(mt.getViterbiNormalizedCharRunning());
			
			//System.out.println("\n" + transcriptionOutputBuffer.toString() + "\n");
			
			if (outputFormats.contains(NORM)) {
				System.out.println("Writing normalized transcription output to " + transcriptionOutputFilename);
				FileHelper.writeString(transcriptionOutputFilename, transcriptionOutputBuffer);
			}
		}
		
		//
		// Make comparison file
		//
		//if ((allowGlyphSubstitution || goldDiplomaticLineChars != null || goldNormalizedLineChars != null) && outputFormats.contains(COMP)) {
		if (outputFormats.contains(COMP)) {
			String transcriptionOutputFilename = comparisonsTranscriptionOutputFile(outputFilenameBase);
			
			List<String> transcriptionWithSubsOutputLines = getTranscriptionLinesWithSubs(mt.getViterbiDecodeStates());
			
			StringBuffer goldComparisonOutputBuffer = new StringBuffer();
			if (allowGlyphSubstitution)          goldComparisonOutputBuffer.append("MN: " + "Model normalized transcription\n");
			if (goldNormalizedLineChars != null) goldComparisonOutputBuffer.append("GN: " + "Gold normalized transcription\n");
			/*                       */          goldComparisonOutputBuffer.append("MD: " + "Model diplomatic transcription\n");
			if (goldDiplomaticLineChars != null) goldComparisonOutputBuffer.append("GD: " + "Gold diplomatic transcription\n");
			if (allowGlyphSubstitution)          goldComparisonOutputBuffer.append("MS: " + "Model transcription with substitutions\n");
			goldComparisonOutputBuffer.append("\n\n");
			
			for (int line = 0; line < numLines; ++line) {
				if (allowGlyphSubstitution)          goldComparisonOutputBuffer.append("MN: " + joinLineForPrinting(mt.getViterbiNormalizedCharLines()[line]).trim() + "\n");
				if (goldNormalizedLineChars != null) goldComparisonOutputBuffer.append("GN: " + joinLineForPrinting(Arrays.asList(goldNormalizedLineChars[line])).trim() + "\n");
				/*                       */          goldComparisonOutputBuffer.append("MD: " + joinLineForPrinting(mt.getViterbiDiplomaticCharLines()[line]).trim() + "\n");
				if (goldDiplomaticLineChars != null) goldComparisonOutputBuffer.append("GD: " + joinLineForPrinting(Arrays.asList(goldDiplomaticLineChars[line])).trim() + "\n");
				if (allowGlyphSubstitution)          goldComparisonOutputBuffer.append("MS: " + transcriptionWithSubsOutputLines.get(line).trim()+"\n");
				goldComparisonOutputBuffer.append("\n");
			}
			goldComparisonOutputBuffer.append("\n");
			
			if ((allowGlyphSubstitution && mt.getViterbiNormalizedCharRunning() != null) || goldNormalizedChars != null) {
				if ((allowGlyphSubstitution && mt.getViterbiNormalizedCharRunning() != null) && goldNormalizedChars != null) {
					goldComparisonOutputBuffer.append("Model (top) vs. Gold (bottom) normalized transcriptions\n");
				}
				else if (allowGlyphSubstitution && mt.getViterbiNormalizedCharRunning() != null) {
					goldComparisonOutputBuffer.append("Model normalized transcription\n");
				}
				else if (goldNormalizedChars != null) {
					goldComparisonOutputBuffer.append("Gold normalized transcription\n");
				}
				
				if (allowGlyphSubstitution && mt.getViterbiNormalizedCharRunning() != null) {
					goldComparisonOutputBuffer.append(joinLineForPrinting(mt.getViterbiNormalizedCharRunning()) + "\n");
				}
				if (goldNormalizedChars != null) {
					goldComparisonOutputBuffer.append(joinLineForPrinting(goldNormalizedChars) + "\n");
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
			f.writeString(transcriptionOutputFilename, goldComparisonOutputBuffer.toString());
		}
		
		double lmPerplexity = 0; // new LmPerplexity(lm).perplexity(mt.viterbiNormalizedTranscriptionCharIndices, mt.viterbiNormalizedTranscriptionLangIndices);
//		System.out.println("LM perplexity = " + lmPerplexity);
		
		//
		// Other files
		//
		if (outputFormats.contains(ALTO)) {
			new AltoOutputWriter(charIndexer, langIndexer).write(numLines, mt.getViterbiDecodeStates(), doc, outputFilenameBase, inputDocPath, commandLineArgs, false, lmPerplexity);
			if (allowGlyphSubstitution) {
				new AltoOutputWriter(charIndexer, langIndexer).write(numLines, mt.getViterbiDecodeStates(), doc, outputFilenameBase, inputDocPath, commandLineArgs, true, lmPerplexity);
			}
		}
		if (outputFormats.contains(HTML)) {
			new HtmlOutputWriter(charIndexer, langIndexer).write(numLines, mt.getViterbiDecodeStates(), doc.baseName(), outputFilenameBase);
		}
		if (outputFormats.contains(WHITESPACE)) {
			StringBuilder whitespaceFileBuf = new StringBuilder();
			Indexer<String> charIndexer = lm.getCharacterIndexer();
			for (List<DecodeState> decodeStateLine : mt.getViterbiDecodeStates()) {
				int whitespace = 0;
				for (DecodeState ds : decodeStateLine) {
					int c = ds.ts.getGlyphChar().templateCharIndex;
					if (c == charIndexer.getIndex(Charset.SPACE)) {
						whitespace += ds.charWidth;
					}
					else {
						if (whitespace > 0) {
							whitespaceFileBuf.append("{" + whitespace + "}");
							whitespace = 0;
						}
						whitespaceFileBuf.append(Charset.unescapeChar(charIndexer.getObject(c)));
					}
					whitespace += ds.padWidth;
				}
				if (whitespace > 0) {
					whitespaceFileBuf.append("{" + whitespace + "}");
				}
				whitespaceFileBuf.append("\n");
			}

			String whitespaceOutputFilename = outputFilenameBase + "_whitespace.txt";
			System.out.println("Writing whitespace layout to " + whitespaceOutputFilename);
			f.writeString(whitespaceOutputFilename, whitespaceFileBuf.toString());
		}

		//
		// Transcription with widths
		//
//		if (allowGlyphSubstitution) {
//		System.out.println("Transcription with widths");
//		StringBuffer transcriptionWithWidthsOutputBuffer = new StringBuffer();
//		for (int line = 0; line < numLines; ++line) {
//			transcriptionWithWidthsOutputBuffer.append(transcriptionWithSubsOutputLines.get(line));
//			for (int i = 0; i < viterbiTrmt.viterbies[line].size(); ++i) {
//				TransitionState ts = viterbiDecodeStates[line].get(i);
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

	private Tuple2<Tuple3<String[][], String[][], List<String>>, DecodeState[][]> loadGoldTranscriptions(Document doc, DecodeState[][] decodeStates) {
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
			DecodeState[][] newDecodeStates = new DecodeState[numLines][];
			for (int line = 0; line < numLines; ++line) {
				newDecodeStates[line] = line < decodeStates.length ? decodeStates[line] : new DecodeState[0];
			}
			decodeStates = newDecodeStates;
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
		
		return Tuple2(Tuple3(goldDiplomaticCharLines, goldNormalizedCharLines, goldNormalizedChars), decodeStates);
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

	private List<String> getTranscriptionLinesWithSubs(List<DecodeState>[] viterbiDecodeStates) {
		List<String> transcriptionWithSubsOutputLines = new ArrayList<String>();
		for (List<DecodeState> lineStates : viterbiDecodeStates) {
			StringBuilder lineBuffer = new StringBuilder();
			for (DecodeState ds : lineStates) {
				TransitionState ts = ds.ts;
				int lmChar = ts.getLmCharIndex();
				GlyphChar glyph = ts.getGlyphChar();
				int glyphChar = glyph.templateCharIndex;
				String sglyphChar = Charset.unescapeChar(charIndexer.getObject(glyphChar));
				if (lmChar != glyphChar || glyph.glyphType != GlyphType.NORMAL_CHAR) {
					String norm = Charset.unescapeChar(charIndexer.getObject(lmChar));
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
