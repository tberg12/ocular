package edu.berkeley.cs.nlp.ocular.model;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.HYPHEN;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.makeTuple2;
import static edu.berkeley.cs.nlp.ocular.util.Tuple3.makeTuple3;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import edu.berkeley.cs.nlp.ocular.data.ImageLoader.Document;
import edu.berkeley.cs.nlp.ocular.data.textreader.Charset;
import edu.berkeley.cs.nlp.ocular.eval.EMDocumentEvaluator;
import edu.berkeley.cs.nlp.ocular.eval.EMIterationEvaluator;
import edu.berkeley.cs.nlp.ocular.eval.Evaluator;
import edu.berkeley.cs.nlp.ocular.eval.Evaluator.EvalSuffStats;
import edu.berkeley.cs.nlp.ocular.lm.BasicCodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.sub.BasicGlyphSubstitutionModel.BasicGlyphSubstitutionModelFactory;
import edu.berkeley.cs.nlp.ocular.sub.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.util.FileHelper;
import edu.berkeley.cs.nlp.ocular.util.StringHelper;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import edu.berkeley.cs.nlp.ocular.util.Tuple3;
import indexer.Indexer;
import threading.BetterThreader;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class FontTrainEM {
	
	private BasicGlyphSubstitutionModelFactory gsmFactory;
	private DecoderEM decoderEM;
	private EMDocumentEvaluator emDocumentEvaluator;

	private Indexer<String> langIndexer;
	private Indexer<String> charIndexer;
	
	private boolean accumulateBatchesWithinIter;
	private int minDocBatchSize;
	private int updateDocBatchSize;
	
	private int numMstepThreads;
	
	private EMIterationEvaluator emEvalSetIterationEvaluator;
	private int evalFreq;
	private boolean evalBatches;

	public FontTrainEM(
			Indexer<String> langIndexer,
			Indexer<String> charIndexer,
			DecoderEM decoderEM,
			BasicGlyphSubstitutionModelFactory gsmFactory,
			EMDocumentEvaluator emDocumentEvaluator,
			boolean accumulateBatchesWithinIter, int minDocBatchSize, int updateDocBatchSize,
			int numMstepThreads,
			EMIterationEvaluator emEvalSetIterationEvaluator, int evalFreq, boolean evalBatches) {
		
		this.langIndexer = langIndexer;
		this.charIndexer = charIndexer;

		this.gsmFactory = gsmFactory;
		this.decoderEM = decoderEM;
		this.emDocumentEvaluator = emDocumentEvaluator;

		this.accumulateBatchesWithinIter = accumulateBatchesWithinIter;
		this.minDocBatchSize = minDocBatchSize;
		this.updateDocBatchSize = updateDocBatchSize;
		this.numMstepThreads = numMstepThreads;
		
		this.emEvalSetIterationEvaluator = emEvalSetIterationEvaluator;
		this.evalFreq = evalFreq;
		this.evalBatches = evalBatches;
	}

	public Tuple3<CodeSwitchLanguageModel, GlyphSubstitutionModel, Map<String, CharacterTemplate>> run(
				List<Document> documents, String inputPath, String outputPath,
				boolean learnFont,
				boolean retrainLM,
				boolean retrainGSM,
				int numEMIters,
				CodeSwitchLanguageModel lm, GlyphSubstitutionModel gsm, Map<String, CharacterTemplate> font) {
		int numUsableDocs = documents.size();
		
		final CharacterTemplate[] templates = loadTemplates(font, charIndexer);

		if (!learnFont) numEMIters = 0;
		else if (numEMIters <= 0) new RuntimeException("If learnFont=true, then numEMIters must be a positive number.");

		//long overallEmissionCacheNanoTime = 0;
		
		for (int iter = 1; (/* learnFont && */ iter <= numEMIters) || (/* !learnFont && */ iter == 1); ++iter) {
			if (learnFont) System.out.println("Training iteration: " + iter + "  (learnFont=true).");
			else System.out.println("Transcribing (learnFont = false).");
			List<Tuple2<String, Map<String, EvalSuffStats>>> allTrainEvals = new ArrayList<Tuple2<String, Map<String, EvalSuffStats>>>();

			DenseBigramTransitionModel backwardTransitionModel = new DenseBigramTransitionModel(lm);

			// The number of characters assigned to a particular language (to re-estimate language probabilities).
			clearTemplates(templates);

			double totalIterationJointLogProb = 0.0;
			double totalBatchJointLogProb = 0.0;
			int completedBatchesInIteration = 0;
			int batchDocsCounter = 0;
			for (int docNum = 0; docNum < numUsableDocs; ++docNum) {
				Document doc = documents.get(docNum);
				if (learnFont) System.out.println("Training iteration "+iter+" of "+numEMIters+", document: "+(docNum+1)+" of "+numUsableDocs+":  "+doc.baseName());
				else System.out.println("Transcribing document: "+docNum+" of "+numUsableDocs+":  "+doc.baseName());
				doc.loadLineText();

				// e-step
				Tuple2<Tuple2<TransitionState[][], int[][]>, Double> decodeResults = decoderEM.computeEStep(doc, learnFont, lm, gsm, templates, backwardTransitionModel);
				final TransitionState[][] decodeStates = decodeResults._1._1;
				final int[][] decodeWidths = decodeResults._1._2;
				totalIterationJointLogProb += decodeResults._2;
				totalBatchJointLogProb += decodeResults._2;

				// evaluate
				emDocumentEvaluator.printTranscriptionWithEvaluation(iter, 0, doc, decodeStates, decodeWidths, learnFont, inputPath, numEMIters, outputPath, allTrainEvals);

				// m-step
				{
				boolean batchComplete = false;
				int trueMinBatchSize = Math.min(minDocBatchSize, updateDocBatchSize); // min batch size may not exceed standard batch size
				if (docNum+1 == numUsableDocs) { // last document of the set
					batchComplete = true;
				}
				else if (numUsableDocs - (docNum+1) < trueMinBatchSize) { // next batch will be too small, so lump the remaining documents in with this one
					// no update
				} 
				else if ((docNum+1) % updateDocBatchSize == 0) { // batch is complete
					batchComplete = true;
				}
				
				if (batchComplete) {
					List<TransitionState> fullViterbiStateSeq = makeFullViterbiStateSeq(decodeStates, charIndexer);
					if (learnFont) updateFontParameters(iter, templates);
					if (retrainLM) lm = updateLmParameters(lm, fullViterbiStateSeq);
					if (retrainGSM) gsm = updateGsmParameters(lm, fullViterbiStateSeq, iter, completedBatchesInIteration);

					++completedBatchesInIteration;
					++batchDocsCounter;
					double avgLogProb = ((double)totalBatchJointLogProb) / batchDocsCounter;
					System.out.println("Iteration "+iter+", batch "+completedBatchesInIteration+": avg joint log prob: " + avgLogProb);
					if (evalBatches) {
						if (iter % evalFreq == 0 || iter == numEMIters) { // evaluate after evalFreq iterations, and at the very end
							if (iter != numEMIters || docNum+1 != numUsableDocs) { // don't evaluate the last batch of the training because it will be done below
								emEvalSetIterationEvaluator.printTranscriptionWithEvaluation(iter, completedBatchesInIteration, lm, gsm, font);
							}
						}
					}
					totalBatchJointLogProb = 0;
					batchDocsCounter = 0;
					}
				} // end: m-step
			} // end: for (doc in usableDocs)
			double avgLogProb = ((double)totalIterationJointLogProb) / numUsableDocs;
			System.out.println("Iteration "+iter+" avg joint log prob: " + avgLogProb);
			if (new File(inputPath).isDirectory()) {
				printEvaluation(allTrainEvals, outputPath + "/" + new File(inputPath).getName() + "/eval_iter-"+iter+".txt");
			}
			
			if (iter % evalFreq == 0 || iter == numEMIters) { // evaluate after evalFreq iterations, and at the very end
				System.out.println("Evaluating dev data at the end of iteration "+iter);
				emEvalSetIterationEvaluator.printTranscriptionWithEvaluation(iter, 0, lm, gsm, font);
			}
		} // end: for iteration
		

		//System.out.println("Emission cache time: " + overallEmissionCacheNanoTime / 1e9 + "s");
		return makeTuple3(lm, gsm, font);
	}

	public static CharacterTemplate[] loadTemplates(Map<String, CharacterTemplate> font, Indexer<String> charIndexer) {
		final CharacterTemplate[] templates = new CharacterTemplate[charIndexer.size()];
		for (int c = 0; c < charIndexer.size(); ++c) {
			CharacterTemplate template = font.get(charIndexer.getObject(c));
			if (template == null)
				throw new RuntimeException("No template found for character '"+charIndexer.getObject(c)+"' ("+StringHelper.toUnicode(charIndexer.getObject(c))+")");
			templates[c] = template;
		}
		return templates;
	}


	private void clearTemplates(final CharacterTemplate[] templates) {
		for (int c = 0; c < templates.length; ++c) {
			if (templates[c] != null) 
				templates[c].clearCounts();
		}
	}

	private void updateFontParameters(int iter, final CharacterTemplate[] templates) {
		long nanoTime = System.nanoTime();
		final int iterFinal = iter;
		BetterThreader.Function<Integer, Object> func = new BetterThreader.Function<Integer, Object>() {
			public void call(Integer c, Object ignore) {
				if (templates[c] != null) templates[c].updateParameters(iterFinal);
			}
		};
		BetterThreader<Integer, Object> threader = new BetterThreader<Integer, Object>(func, numMstepThreads);
		for (int c = 0; c < templates.length; ++c)
			threader.addFunctionArgument(c);
		threader.run();
		System.out.println("Update font parameters: " + (System.nanoTime() - nanoTime) / 1000000 + "ms");
		
		if (!accumulateBatchesWithinIter) {
			System.out.println("Clearing font parameter statistics.");
			clearTemplates(templates);
		}
	}

	/**
	 * Hard-EM update on language model probabilities
	 */
	private CodeSwitchLanguageModel updateLmParameters(CodeSwitchLanguageModel lm, List<TransitionState> fullViterbiStateSeq) {
		long nanoTime = System.nanoTime();
		
		//
		// Initialize containers for counts
		//
		int numLanguages = langIndexer.size();
		int[] languageCounts = new int[numLanguages];
		Arrays.fill(languageCounts, 1); // one-count smooth
		
		//
		// Pass over the decoded states to accumulate counts
		//
		for (TransitionState ts : fullViterbiStateSeq) {
			int currLanguage = ts.getLanguageIndex();
			if (currLanguage >= 0) { 
				languageCounts[currLanguage] += 1;
			}
		}
		
		//
		// Update the parameters using counts
		//
		List<Tuple2<SingleLanguageModel, Double>> newSubModelsAndPriors = new ArrayList<Tuple2<SingleLanguageModel, Double>>();
		double languageCountSum = 0;
		for (int language = 0; language < numLanguages; ++language) {
			double newPrior = languageCounts[language];
			newSubModelsAndPriors.add(makeTuple2(lm.get(language), newPrior));
			languageCountSum += newPrior;
		}

		//
		// Construct the new LM
		//
		CodeSwitchLanguageModel newLM = new BasicCodeSwitchLanguageModel(newSubModelsAndPriors, lm.getCharacterIndexer(), langIndexer, lm.getProbKeepSameLanguage(), lm.getMaxOrder());

		//
		// Print out some statistics
		//
		StringBuilder sb = new StringBuilder("Updating language probabilities: ");
		for (int language = 0; language < numLanguages; ++language)
			sb.append(langIndexer.getObject(language)).append("->").append(languageCounts[language] / languageCountSum).append("  ");
		System.out.println(sb);
		
		System.out.println("New LM: " + (System.nanoTime() - nanoTime) / 1000000 + "ms");
		return newLM;
	}

	/**
	 * Hard-EM update on glyph substitution probabilities
	 */
	private GlyphSubstitutionModel updateGsmParameters(CodeSwitchLanguageModel newLM, List<TransitionState> fullViterbiStateSeq, int iter, int batchId) {
		long nanoTime = System.nanoTime();

		//
		// Construct the new GSM
		//
		GlyphSubstitutionModel newGSM = gsmFactory.make(fullViterbiStateSeq, newLM, iter, batchId);

		//
		// Print out some statistics
		//
		System.out.println("New GSM: " + (System.nanoTime() - nanoTime) / 1000000 + "ms");
		return newGSM;
	}

	/**
	 * Make a single sequence of states
	 */
	public static List<TransitionState> makeFullViterbiStateSeq(TransitionState[][] decodeStates, Indexer<String> charIndexer) {
		int spaceIndex = charIndexer.getIndex(Charset.SPACE);
		int hyphenIndex = charIndexer.getIndex(Charset.HYPHEN);
		int numLines = decodeStates.length;
		@SuppressWarnings("unchecked")
		List<String>[] viterbiChars = new List[numLines];
		List<TransitionState> fullViterbiStateSeq = new ArrayList<TransitionState>();
		for (int line = 0; line < numLines; ++line) {
			viterbiChars[line] = new ArrayList<String>();
			if (line < decodeStates.length) {
				int lineLength = decodeStates[line].length;
				
				// Figure out stuff about the end of the line hyphen whatever
				int lastCharOfLine = -1;
				boolean endsInRightHyphen = false;
				for (int i = 0; i < lineLength; ++i) {
					TransitionState ts = decodeStates[line][i];
					TransitionStateType tsType = ts.getType();
					if (tsType == TransitionStateType.TMPL && ts.getGlyphChar().templateCharIndex != spaceIndex) {
						lastCharOfLine = i;
					}
					endsInRightHyphen = (tsType == TransitionStateType.RMRGN_HPHN_INIT || tsType == TransitionStateType.RMRGN_HPHN);
				}
				
				// Handle all the states
				for (int i = 0; i < lineLength; ++i) {
					TransitionState ts = decodeStates[line][i];
					int c = ts.getGlyphChar().templateCharIndex;
					if (viterbiChars[line].isEmpty() || !(HYPHEN.equals(viterbiChars[line].get(viterbiChars[line].size() - 1)) && HYPHEN.equals(charIndexer.getObject(c)))) {
						viterbiChars[line].add(charIndexer.getObject(c));
						if (ts.getType() == TransitionStateType.TMPL) {
							if (c != hyphenIndex || !endsInRightHyphen || i != lastCharOfLine) {
								fullViterbiStateSeq.add(ts);
							}
						}
					}
				}
			}
		}
		return fullViterbiStateSeq;
	}

	public static void printEvaluation(List<Tuple2<String, Map<String, EvalSuffStats>>> allEvals, String outputPath) {
		Map<String, EvalSuffStats> totalSuffStats = new HashMap<String, EvalSuffStats>();
		StringBuffer buf = new StringBuffer();
		buf.append("All evals:\n");
		for (Tuple2<String, Map<String, EvalSuffStats>> docNameAndEvals : allEvals) {
			String docName = docNameAndEvals._1;
			Map<String, EvalSuffStats> evals = docNameAndEvals._2;
			buf.append("Document: " + docName + "\n");
			buf.append(Evaluator.renderEval(evals) + "\n");
			for (String evalType : evals.keySet()) {
				EvalSuffStats eval = evals.get(evalType);
				EvalSuffStats totalEval = totalSuffStats.get(evalType);
				if (totalEval == null) {
					totalEval = new EvalSuffStats();
					totalSuffStats.put(evalType, totalEval);
				}
				totalEval.increment(eval);
			}
		}

		buf.append("\nMarco-avg total eval:\n");
		buf.append(Evaluator.renderEval(totalSuffStats) + "\n");

		FileHelper.writeString(outputPath, buf.toString());
		System.out.println("\n" + outputPath);
		System.out.println(buf.toString());
	}

}
