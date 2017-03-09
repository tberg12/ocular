package edu.berkeley.cs.nlp.ocular.train;

import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.HYPHEN;
import static edu.berkeley.cs.nlp.ocular.data.textreader.Charset.SPACE;
import static edu.berkeley.cs.nlp.ocular.eval.EvalPrinter.printEvaluation;
import static edu.berkeley.cs.nlp.ocular.train.ModelPathMaker.makeFontPath;
import static edu.berkeley.cs.nlp.ocular.train.ModelPathMaker.makeGsmPath;
import static edu.berkeley.cs.nlp.ocular.train.ModelPathMaker.makeLmPath;
import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;
import static edu.berkeley.cs.nlp.ocular.util.Tuple3.Tuple3;

import java.io.File;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Calendar;
import java.util.List;
import java.util.Map;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.data.Document;
import edu.berkeley.cs.nlp.ocular.eval.Evaluator.EvalSuffStats;
import edu.berkeley.cs.nlp.ocular.eval.ModelTranscriptions;
import edu.berkeley.cs.nlp.ocular.eval.MultiDocumentTranscriber;
import edu.berkeley.cs.nlp.ocular.eval.SingleDocumentEvaluatorAndOutputPrinter;
import edu.berkeley.cs.nlp.ocular.font.Font;
import edu.berkeley.cs.nlp.ocular.gsm.BasicGlyphSubstitutionModel.BasicGlyphSubstitutionModelFactory;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.lm.BasicCodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.CorpusCounter;
import edu.berkeley.cs.nlp.ocular.lm.InterpolatingSingleLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.NgramLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.NgramLanguageModel.LMType;
import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;
import edu.berkeley.cs.nlp.ocular.main.FonttrainTranscribeShared.OutputFormat;
import edu.berkeley.cs.nlp.ocular.main.InitializeFont;
import edu.berkeley.cs.nlp.ocular.main.InitializeGlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.main.InitializeLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import edu.berkeley.cs.nlp.ocular.model.DecodeState;
import edu.berkeley.cs.nlp.ocular.model.DecoderEM;
import edu.berkeley.cs.nlp.ocular.model.TransitionStateType;
import edu.berkeley.cs.nlp.ocular.model.em.DenseBigramTransitionModel;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.util.CollectionHelper;
import edu.berkeley.cs.nlp.ocular.util.StringHelper;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import edu.berkeley.cs.nlp.ocular.util.Tuple3;
import tberg.murphy.indexer.Indexer;
import tberg.murphy.threading.BetterThreader;

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
				String inputDocPath, String outputPath, Set<OutputFormat> outputFormats,
				MultiDocumentTranscriber evalSetIterationEvaluator, int evalFreq, boolean evalBatches,
				boolean skipFailedDocs) {
		
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
									inputDocPath, outputPath, outputFormats,
									evalSetIterationEvaluator, evalFreq, evalBatches,
									skipFailedDocs);
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
			String inputDocPath, String outputPath, Set<OutputFormat> outputFormats,
			MultiDocumentTranscriber evalSetIterationEvaluator, int evalFreq, boolean evalBatches,
			boolean skipFailedDocs) {
		
		Indexer<String> charIndexer = lm.getCharacterIndexer();
		Indexer<String> langIndexer = lm.getLanguageIndexer();
		int numLanguages = langIndexer.size();
		int numUsableDocs = inputDocuments.size();

		// Containers for counts that will be accumulated
		final CharacterTemplate[] templates = loadTemplates(font, charIndexer);
		List<DecodeState[][]> accumulatedTranscriptions; // for re-estimating the LM
		double[][][] gsmCounts;

		List<Tuple2<String, Map<String, EvalSuffStats>>> allDiplomaticTrainEvals = new ArrayList<Tuple2<String, Map<String, EvalSuffStats>>>();
		List<Tuple2<String, Map<String, EvalSuffStats>>> allNormalizedTrainEvals = new ArrayList<Tuple2<String, Map<String, EvalSuffStats>>>();

		DenseBigramTransitionModel backwardTransitionModel = new DenseBigramTransitionModel(lm);

		// Clear counts at the start of the iteration
		clearTemplates(templates);
		accumulatedTranscriptions = new ArrayList<DecodeState[][]>();
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
			try {
				Tuple2<DecodeState[][], Double> decodeResults = decoderEM.computeEStep(doc, true, lm, gsm, templates, backwardTransitionModel);
				final DecodeState[][] decodeStates = decodeResults._1;
				totalIterationJointLogProb += decodeResults._2;
				totalBatchJointLogProb += decodeResults._2;
				accumulatedTranscriptions.add(decodeStates);
				List<DecodeState> fullViterbiStateSeq = makeFullViterbiStateSeq(decodeStates, charIndexer);
				gsmFactory.incrementCounts(gsmCounts, fullViterbiStateSeq);
				
				// write transcriptions and evaluate
				Tuple2<Map<String, EvalSuffStats>,Map<String, EvalSuffStats>> evals = documentEvaluatorAndOutputPrinter.evaluateAndPrintTranscription(iter, 0, doc, decodeStates, inputDocPath, outputPath, outputFormats, lm);
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
						double newDataInterpolationWeight = 0.5;
						lm = reestimateLM(accumulatedTranscriptions, lm, newDataInterpolationWeight);
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
					accumulatedTranscriptions = new ArrayList<DecodeState[][]>();
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
			} catch(RuntimeException e) {
				if (skipFailedDocs) {
					System.err.println("DOCUMENT FAILED! Skipping " + doc.baseName());
					e.printStackTrace();
				} else {
					throw e;
				}
			}
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
	private void incrementLmCounts(int[] languageCounts, List<DecodeState> fullViterbiStateSeq, Indexer<String> charIndexer) {
		int spaceCharIndex = charIndexer.getIndex(SPACE);
		for (DecodeState ds : fullViterbiStateSeq) {
			TransitionState ts = ds.ts;
			int currLanguage = ts.getLanguageIndex();
			if (currLanguage >= 0 && ts.getType() == TransitionStateType.TMPL && ts.getLmCharIndex() != spaceCharIndex) { 
				languageCounts[currLanguage] += 1;
			}
		}
	}
	
	/**
	 * Hard-EM update on language model probabilities
	 * 
	 * TODO Increase newDataInterpolationWeight on each interation.
	 * TODO Use lower newDataInterpolationWeight if totalChars for the language is too low.
	 */
	private CodeSwitchLanguageModel reestimateLM(List<DecodeState[][]> accumulatedTranscriptions, CodeSwitchLanguageModel cslm, double newDataInterpolationWeight) {
		long nanoTime = System.nanoTime();
		Indexer<String> charIndexer = cslm.getCharacterIndexer();
		Indexer<String> langIndexer = cslm.getLanguageIndexer();
		int numLangs = langIndexer.size();
		
		System.out.println("Retraining LM");
		List<List<List<String>>> allTranscriptionsByLanguage = separateTranscriptionsByLanguage(accumulatedTranscriptions, charIndexer, langIndexer);
		
		List<Tuple2<SingleLanguageModel, Double>> lmsAndPriors = new ArrayList<Tuple2<SingleLanguageModel, Double>>();
		for (int langIndex = 0; langIndex < numLangs; ++langIndex) {
			SingleLanguageModel langBaseLM = cslm.get(langIndex);
			if (langBaseLM instanceof InterpolatingSingleLanguageModel) {
				langBaseLM = ((InterpolatingSingleLanguageModel)langBaseLM).getSubModel(0);
			}
			int ngramLength = langBaseLM.getMaxOrder();
			CorpusCounter counter = new CorpusCounter(ngramLength);
			int totalLangChars = 0;
			for (List<String> chars : allTranscriptionsByLanguage.get(langIndex)) {
				counter.countChars(chars, charIndexer, 0);
				totalLangChars += chars.size();
			}
			System.out.println("  found " + totalLangChars + " characters for " + langIndexer.getObject(langIndex) + " read from transcription output");
//			for (List<String> chars : allTranscriptionsByLanguage.get(langIndex)) {
//				System.err.println("    " + StringHelper.join(chars));
//			}
			
			SingleLanguageModel langUpdatedLM;
			if (totalLangChars > 0) {
				double lmPower = (langBaseLM instanceof NgramLanguageModel ? ((NgramLanguageModel)langBaseLM).getLmPower() : 4.0);
				SingleLanguageModel langNewLM = new NgramLanguageModel(charIndexer, counter.getCounts(), langBaseLM.getActiveCharacters(), LMType.KNESER_NEY, lmPower);
				langUpdatedLM = new InterpolatingSingleLanguageModel(CollectionHelper.makeList(
						Tuple2(langBaseLM, 1.0 - newDataInterpolationWeight), Tuple2(langNewLM, newDataInterpolationWeight)));
				System.out.println("  using new interpolated lm for " + langIndexer.getObject(langIndex));
			}
			else {
				System.out.println("  using original lm for " + langIndexer.getObject(langIndex));
				langUpdatedLM = langBaseLM; // this language doesn't appear in the transcription, so nothing to interpolate with.
			}
			double prior = totalLangChars + 1.0; // for smoothing
			lmsAndPriors.add(Tuple2(langUpdatedLM, prior)); // prior gets normalized later
		}
		CodeSwitchLanguageModel newCodeSwitchLM = new BasicCodeSwitchLanguageModel(lmsAndPriors, charIndexer, langIndexer, cslm.getProbKeepSameLanguage());
		
		System.out.println("New LM: " + (System.nanoTime() - nanoTime) / 1000000 + "ms");
		return newCodeSwitchLM;
	}
	
	private List<List<List<String>>> separateTranscriptionsByLanguage(List<DecodeState[][]> accumulatedTranscriptions, Indexer<String> charIndexer, Indexer<String> langIndexer) {
		int numLangs = langIndexer.size();
		
		// For each language, we will have a list of passages.
		List<List<List<String>>> allTranscriptionsByLanguage = new ArrayList<List<List<String>>>();
		for (int i = 0; i < numLangs; ++i) {
			allTranscriptionsByLanguage.add(new ArrayList<List<String>>());
		}
		
		for (DecodeState[][] decodeStates : accumulatedTranscriptions) {
			ModelTranscriptions mt = new ModelTranscriptions(decodeStates, charIndexer, langIndexer);
			List<Tuple2<String, String>> normalizedCharLangTranscription = mt.getViterbiNormalizedCharLangRunning();
			
			String prevLanguage = null;
			List<String> currentOutput = new ArrayList<String>();
			for (Tuple2<String, String> charLang : normalizedCharLangTranscription) {
				String currLanguage = charLang._2;
				if (!equalsNullSafe(currLanguage, prevLanguage)) {
					if (!currentOutput.isEmpty()) {
						int prevLangIndex = (prevLanguage == null && numLangs == 1 ? 0 : langIndexer.getIndex(prevLanguage));
						allTranscriptionsByLanguage.get(prevLangIndex).add(currentOutput);
					}
					currentOutput = new ArrayList<String>();
					prevLanguage = currLanguage;
				}
				if (!SPACE.equals(charLang._1) || currentOutput.isEmpty() || !SPACE.equals(CollectionHelper.last(currentOutput))) { // don't put two spaces in a row
					currentOutput.add(charLang._1);
				}
			}
			if (!currentOutput.isEmpty()) {
				int prevLangIndex = (prevLanguage == null && numLangs == 1 ? 0 : langIndexer.getIndex(prevLanguage));
				allTranscriptionsByLanguage.get(prevLangIndex).add(currentOutput);
			}
		}
		
		return allTranscriptionsByLanguage;
	}

	private <A> boolean equalsNullSafe(A o1, A o2) {
		if (o1 == null && o2 == null)
			return true;
		if (o1 == null || o2 == null)
			return false;
		return o1.equals(o2);
	}

	/**
	 * Make a single sequence of states
	 */
	public static List<DecodeState> makeFullViterbiStateSeq(DecodeState[][] decodeStates, Indexer<String> charIndexer) {
		int numLines = decodeStates.length;
		@SuppressWarnings("unchecked")
		List<String>[] viterbiChars = new List[numLines];
		List<DecodeState> fullViterbiStateSeq = new ArrayList<DecodeState>();
		for (int line = 0; line < numLines; ++line) {
			viterbiChars[line] = new ArrayList<String>();
			if (line < decodeStates.length) {
				int lineLength = decodeStates[line].length;
				
				// Handle all the states
				for (int i = 0; i < lineLength; ++i) {
					DecodeState decodeState = decodeStates[line][i];
					int c = decodeState.ts.getGlyphChar().templateCharIndex;
					if (viterbiChars[line].isEmpty() || !(HYPHEN.equals(viterbiChars[line].get(viterbiChars[line].size() - 1)) && HYPHEN.equals(charIndexer.getObject(c)))) {
						viterbiChars[line].add(charIndexer.getObject(c));
						fullViterbiStateSeq.add(decodeState);
					}
				}
			}
		}
		return fullViterbiStateSeq;
	}

}
