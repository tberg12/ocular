package edu.berkeley.cs.nlp.ocular.train;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.HYPHEN;
import static edu.berkeley.cs.nlp.ocular.eval.EvalPrinter.printEvaluation;
import static edu.berkeley.cs.nlp.ocular.train.ModelPathMaker.makeFontPath;
import static edu.berkeley.cs.nlp.ocular.train.ModelPathMaker.makeGsmPath;
import static edu.berkeley.cs.nlp.ocular.train.ModelPathMaker.makeLmPath;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;
import static edu.berkeley.cs.nlp.ocular.util.Tuple3.Tuple3;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.List;
import java.util.Map;

import edu.berkeley.cs.nlp.ocular.data.Document;
import edu.berkeley.cs.nlp.ocular.eval.Evaluator.EvalSuffStats;
import edu.berkeley.cs.nlp.ocular.eval.MultiDocumentTranscriber;
import edu.berkeley.cs.nlp.ocular.eval.SingleDocumentEvaluatorAndOutputPrinter;
import edu.berkeley.cs.nlp.ocular.font.Font;
import edu.berkeley.cs.nlp.ocular.gsm.BasicGlyphSubstitutionModel.BasicGlyphSubstitutionModelFactory;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.lm.BasicCodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;
import edu.berkeley.cs.nlp.ocular.main.InitializeFont;
import edu.berkeley.cs.nlp.ocular.main.InitializeGlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.main.InitializeLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import edu.berkeley.cs.nlp.ocular.model.DecoderEM;
import edu.berkeley.cs.nlp.ocular.model.TransitionStateType;
import edu.berkeley.cs.nlp.ocular.model.em.DenseBigramTransitionModel;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.util.StringHelper;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import edu.berkeley.cs.nlp.ocular.util.Tuple3;
import indexer.Indexer;
import threading.BetterThreader;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class FontTrainer {
	
	public Tuple3<Font, CodeSwitchLanguageModel, GlyphSubstitutionModel> trainFont(
				List<Document> inputDocuments,  
				Font font, CodeSwitchLanguageModel lm, GlyphSubstitutionModel gsm,
				TrainingRestarter trainingRestarter,
				String outputFontPath, String outputLmPath, String outputGsmPath,
				DecoderEM decoderEM,
				BasicGlyphSubstitutionModelFactory gsmFactory,
				SingleDocumentEvaluatorAndOutputPrinter documentEvaluatorAndOutputPrinter,
				int numEMIters, int updateDocBatchSize, boolean noUpdateIfBatchTooSmall, boolean writeIntermediateModelsToTemp,
				int numMstepThreads,
				String inputDocPath, String outputPath,
				MultiDocumentTranscriber evalSetIterationEvaluator, int evalFreq, boolean evalBatches) {
		
		System.out.println("trainFont(numEMIters="+numEMIters+", updateDocBatchSize="+updateDocBatchSize+", noUpdateIfBatchTooSmall="+noUpdateIfBatchTooSmall+", writeIntermediateModelsToTemp="+writeIntermediateModelsToTemp+")");
		
		int numUsableDocs = inputDocuments.size();

		int lastCompletedIteration = 0;
		Tuple2<Integer, Tuple3<Font,CodeSwitchLanguageModel,GlyphSubstitutionModel>> restartObjects;
		if (trainingRestarter != null) {
			restartObjects = trainingRestarter.getRestartModels(
					font, lm, gsm,
					outputLmPath != null, outputGsmPath != null, outputPath, 
					numEMIters, numUsableDocs, updateDocBatchSize, noUpdateIfBatchTooSmall);
			lastCompletedIteration = restartObjects._1;
			font = restartObjects._2._1;
			lm = restartObjects._2._2;
			gsm = restartObjects._2._3;
		}
		
		//long overallEmissionCacheNanoTime = 0;
		
		for (int iter = lastCompletedIteration+1; (/* trainFont && */ iter <= numEMIters) || (/* !trainFont && */ iter == 1); ++iter) {
			System.out.println("Training iteration: " + iter + "    " + (new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime())));
			
			Tuple3<Font, CodeSwitchLanguageModel, GlyphSubstitutionModel> iterationResultModels = 
					doFontTrainPass(iter,
									inputDocuments,
									font, lm, gsm,
									outputFontPath, outputLmPath, outputGsmPath,
									decoderEM,
									gsmFactory,
									documentEvaluatorAndOutputPrinter,
									numEMIters, updateDocBatchSize, noUpdateIfBatchTooSmall, writeIntermediateModelsToTemp,
									numMstepThreads,
									inputDocPath, outputPath,
									evalSetIterationEvaluator, evalFreq, evalBatches);
			font = iterationResultModels._1;
			lm = iterationResultModels._2;
			gsm = iterationResultModels._3;
			
		} // end: for iteration
		

		//System.out.println("Emission cache time: " + overallEmissionCacheNanoTime / 1e9 + "s");

		//
		// Write final models
		//
		System.out.println("Training completed; saving models.");
		if (outputFontPath != null) {
			System.out.println("Writing trained font to " + outputFontPath);
			InitializeFont.writeFont(font, outputFontPath);
		}
		if (outputLmPath != null) {
			System.out.println("Writing trained lm to " + outputLmPath);
			InitializeLanguageModel.writeLM(lm, outputLmPath);
		}
		if (outputGsmPath != null) {
			System.out.println("Writing trained gsm to " + outputGsmPath);
			InitializeGlyphSubstitutionModel.writeGSM(gsm, outputGsmPath);
		}

		return Tuple3(font, lm, gsm);
	}
	
	
	public Tuple3<Font, CodeSwitchLanguageModel, GlyphSubstitutionModel> doFontTrainPass(
			int iter,
			List<Document> inputDocuments,  
			Font font, CodeSwitchLanguageModel lm, GlyphSubstitutionModel gsm,
			String outputFontPath, String outputLmPath, String outputGsmPath,
			DecoderEM decoderEM,
			BasicGlyphSubstitutionModelFactory gsmFactory,
			SingleDocumentEvaluatorAndOutputPrinter documentEvaluatorAndOutputPrinter,
			int numEMIters, int updateDocBatchSize, boolean noUpdateIfBatchTooSmall, boolean writeIntermediateModelsToTemp,
			int numMstepThreads,
			String inputDocPath, String outputPath,
			MultiDocumentTranscriber evalSetIterationEvaluator, int evalFreq, boolean evalBatches) {
		
		Indexer<String> charIndexer = lm.getCharacterIndexer();
		Indexer<String> langIndexer = lm.getLanguageIndexer();
		int numLanguages = langIndexer.size();
		int numUsableDocs = inputDocuments.size();

		// Containers for counts that will be accumulated
		final CharacterTemplate[] templates = loadTemplates(font, charIndexer);
		int[] languageCounts = new int[numLanguages]; // The number of characters assigned to a particular language (to re-estimate language probabilities).
		double[][][] gsmCounts;

		List<Tuple2<String, Map<String, EvalSuffStats>>> allDiplomaticTrainEvals = new ArrayList<Tuple2<String, Map<String, EvalSuffStats>>>();
		List<Tuple2<String, Map<String, EvalSuffStats>>> allNormalizedTrainEvals = new ArrayList<Tuple2<String, Map<String, EvalSuffStats>>>();

		DenseBigramTransitionModel backwardTransitionModel = new DenseBigramTransitionModel(lm);

		// Clear counts at the start of the iteration
		clearTemplates(templates);
		Arrays.fill(languageCounts, 1); // one-count smooth
		gsmCounts = gsmFactory.initializeNewCountsMatrix();

		double totalIterationJointLogProb = 0.0;
		double totalBatchJointLogProb = 0.0;
		int completedBatchesInIteration = 0;
		int batchDocsCounter = 0;
		for (int docNum = 0; docNum < numUsableDocs; ++docNum) {
			++batchDocsCounter;
			
			Document doc = inputDocuments.get(docNum);
			System.out.println("Training iteration "+iter+" of "+numEMIters+", document "+(docNum+1)+" of "+numUsableDocs+":  "+doc.baseName() + "    " + (new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime())));
			doc.loadDiplomaticTextLines();
			doc.loadNormalizedText();

			// e-step
			Tuple2<Tuple2<TransitionState[][], int[][]>, Double> decodeResults = decoderEM.computeEStep(doc, true, lm, gsm, templates, backwardTransitionModel);
			final TransitionState[][] decodeStates = decodeResults._1._1;
			final int[][] decodeWidths = decodeResults._1._2;
			totalIterationJointLogProb += decodeResults._2;
			totalBatchJointLogProb += decodeResults._2;
			List<TransitionState> fullViterbiStateSeq = makeFullViterbiStateSeq(decodeStates, charIndexer);
			incrementLmCounts(languageCounts, fullViterbiStateSeq, charIndexer);
			gsmFactory.incrementCounts(gsmCounts, fullViterbiStateSeq);
			
			// write transcriptions and evaluate
			Tuple2<Map<String, EvalSuffStats>,Map<String, EvalSuffStats>> evals = documentEvaluatorAndOutputPrinter.evaluateAndPrintTranscription(iter, 0, doc, decodeStates, decodeWidths, inputDocPath, outputPath);
			if (evals._1 != null) allDiplomaticTrainEvals.add(Tuple2(doc.baseName(), evals._1));
			if (evals._2 != null) allNormalizedTrainEvals.add(Tuple2(doc.baseName(), evals._2));
			
			// m-step
			if (isBatchComplete(numUsableDocs, docNum, batchDocsCounter, updateDocBatchSize, noUpdateIfBatchTooSmall)) {
				++completedBatchesInIteration;
				
				if (outputFontPath != null) {
					updateFontParameters(templates, numMstepThreads);
					String writePath = writeIntermediateModelsToTemp ? makeFontPath(outputPath, iter, completedBatchesInIteration) : outputFontPath;
					System.out.println("Writing updated font to " + writePath);
					InitializeFont.writeFont(font, writePath);
				}
				if (outputLmPath != null) {
					lm = reestimateLM(languageCounts, lm);
					String writePath = writeIntermediateModelsToTemp ? makeLmPath(outputPath, iter, completedBatchesInIteration) : outputLmPath;
					System.out.println("Writing updated lm to " + writePath);
					InitializeLanguageModel.writeLM(lm, writePath);
					backwardTransitionModel = new DenseBigramTransitionModel(lm);
				}
				if (outputGsmPath != null) {
					System.out.println("Estimating parameters of a new Glyph Substitution Model.  Iter: "+iter+", batch: "+completedBatchesInIteration);
					gsm = gsmFactory.make(gsmCounts, iter, completedBatchesInIteration);
					String writePath = writeIntermediateModelsToTemp ? makeGsmPath(outputPath, iter, completedBatchesInIteration) : outputGsmPath;
					System.out.println("Writing updated gsm to " + writePath);
					InitializeGlyphSubstitutionModel.writeGSM(gsm, writePath);
				}

				// Clear counts at the end of a batch
				System.out.println("Clearing font parameter statistics.");
				clearTemplates(templates);
				Arrays.fill(languageCounts, 1); // one-count smooth
				gsmCounts = gsmFactory.initializeNewCountsMatrix();
				
				double avgLogProb = ((double)totalBatchJointLogProb) / batchDocsCounter;
				System.out.println("Completed Batch: Iteration "+iter+", batch "+completedBatchesInIteration+": avg joint log prob: " + avgLogProb + "    " + (new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime())));
				if (evalBatches) {
					if (iter % evalFreq == 0 || iter == numEMIters) { // evaluate after evalFreq iterations, and at the very end
						if (iter != numEMIters || docNum+1 != numUsableDocs) { // don't evaluate the last batch of the training because it will be done below
							evalSetIterationEvaluator.transcribe(iter, completedBatchesInIteration, font, lm, gsm);
						}
					}
				}
				totalBatchJointLogProb = 0;
				batchDocsCounter = 0;
			} // end: m-step
		} // end: for (doc in usableDocs)
		
		// evaluate on training data
		double avgLogProb = ((double)totalIterationJointLogProb) / numUsableDocs;
		System.out.println("Iteration "+iter+" avg joint log prob: " + avgLogProb);
		if (new File(inputDocPath).isDirectory()) {
			if (!allDiplomaticTrainEvals.isEmpty())
				printEvaluation(allDiplomaticTrainEvals, outputPath + "/all_transcriptions/" + new File(inputDocPath).getName() + "/eval_iter-"+iter+"_diplomatic.txt");
			if (!allNormalizedTrainEvals.isEmpty())
				printEvaluation(allNormalizedTrainEvals, outputPath + "/all_transcriptions/" + new File(inputDocPath).getName() + "/eval_iter-"+iter+"_normalized.txt");
		}
		
		// evaluate on dev data, if requested
		if (iter % evalFreq == 0 || iter == numEMIters) { // evaluate after evalFreq iterations, and at the very end
			System.out.println("Evaluating dev data at the end of iteration "+iter+"    " + (new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime())));
			evalSetIterationEvaluator.transcribe(iter, 0, font, lm, gsm);
		}
		
		return Tuple3(font, lm, gsm);
	}
	

	public static boolean isBatchComplete(int numUsableDocs, int docNum, int currentBatchSize, int updateDocBatchSize, boolean noUpdateIfBatchTooSmall) {
		boolean batchComplete = false;
		if (docNum+1 == numUsableDocs) { // last document of the set
			if (!noUpdateIfBatchTooSmall || currentBatchSize >= updateDocBatchSize)
				batchComplete = true;
		}
		else if (numUsableDocs - (docNum+1) < updateDocBatchSize) { // next batch will be too small, so lump the remaining documents in with this one
			// no update
		} 
		else if (currentBatchSize == updateDocBatchSize) { // batch is complete
			batchComplete = true;
		}
		return batchComplete;
	}

	public static CharacterTemplate[] loadTemplates(Font font, Indexer<String> charIndexer) {
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

	private void updateFontParameters(final CharacterTemplate[] templates, int numMstepThreads) {
		long nanoTime = System.nanoTime();
		BetterThreader.Function<Integer, Object> func = new BetterThreader.Function<Integer, Object>() {
			public void call(Integer c, Object ignore) {
				if (templates[c] != null) templates[c].updateParameters();
			}
		};
		BetterThreader<Integer, Object> threader = new BetterThreader<Integer, Object>(func, numMstepThreads);
		for (int c = 0; c < templates.length; ++c)
			threader.addFunctionArgument(c);
		threader.run();
		System.out.println("Update font parameters: " + (System.nanoTime() - nanoTime) / 1000000 + "ms");
	}
	
	/**
	 * Pass over the decoded states to accumulate counts
	 */
	private void incrementLmCounts(int[] languageCounts, List<TransitionState> fullViterbiStateSeq, Indexer<String> charIndexer) {
		int spaceCharIndex = charIndexer.getIndex(" ");
		for (TransitionState ts : fullViterbiStateSeq) {
			int currLanguage = ts.getLanguageIndex();
			if (currLanguage >= 0 && ts.getType() == TransitionStateType.TMPL && ts.getLmCharIndex() != spaceCharIndex) { 
				languageCounts[currLanguage] += 1;
			}
		}
	}
	
	/**
	 * Hard-EM update on language model probabilities
	 */
	private CodeSwitchLanguageModel reestimateLM(int[] languageCounts, CodeSwitchLanguageModel lm) {
		long nanoTime = System.nanoTime();
		
		// Update the parameters using counts
		List<Tuple2<SingleLanguageModel, Double>> newSubModelsAndPriors = new ArrayList<Tuple2<SingleLanguageModel, Double>>();
		double languageCountSum = 0;
		for (int language = 0; language < languageCounts.length; ++language) {
			double newPrior = languageCounts[language];
			newSubModelsAndPriors.add(Tuple2(lm.get(language), newPrior));
			languageCountSum += newPrior;
		}

		// Construct the new LM
		CodeSwitchLanguageModel newLM = new BasicCodeSwitchLanguageModel(newSubModelsAndPriors, lm.getCharacterIndexer(), lm.getLanguageIndexer(), lm.getProbKeepSameLanguage(), lm.getMaxOrder());

		// Print out some statistics
		StringBuilder sb = new StringBuilder("Updating language probabilities: ");
		for (int language = 0; language < languageCounts.length; ++language)
			sb.append(lm.getLanguageIndexer().getObject(language)).append("->").append(languageCounts[language] / languageCountSum).append("  ");
		System.out.println(sb);
		
		System.out.println("New LM: " + (System.nanoTime() - nanoTime) / 1000000 + "ms");
		return newLM;
	}

	/**
	 * Make a single sequence of states
	 */
	public static List<TransitionState> makeFullViterbiStateSeq(TransitionState[][] decodeStates, Indexer<String> charIndexer) {
		int numLines = decodeStates.length;
		@SuppressWarnings("unchecked")
		List<String>[] viterbiChars = new List[numLines];
		List<TransitionState> fullViterbiStateSeq = new ArrayList<TransitionState>();
		for (int line = 0; line < numLines; ++line) {
			viterbiChars[line] = new ArrayList<String>();
			if (line < decodeStates.length) {
				int lineLength = decodeStates[line].length;
				
				// Handle all the states
				for (int i = 0; i < lineLength; ++i) {
					TransitionState ts = decodeStates[line][i];
					int c = ts.getGlyphChar().templateCharIndex;
					if (viterbiChars[line].isEmpty() || !(HYPHEN.equals(viterbiChars[line].get(viterbiChars[line].size() - 1)) && HYPHEN.equals(charIndexer.getObject(c)))) {
						viterbiChars[line].add(charIndexer.getObject(c));
						fullViterbiStateSeq.add(ts);
					}
				}
			}
		}
		return fullViterbiStateSeq;
	}

}
