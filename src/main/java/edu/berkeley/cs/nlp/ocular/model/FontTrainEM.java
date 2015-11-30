package edu.berkeley.cs.nlp.ocular.model;

import static edu.berkeley.cs.nlp.ocular.util.Tuple2.makeTuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import edu.berkeley.cs.nlp.ocular.data.ImageLoader.Document;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import edu.berkeley.cs.nlp.ocular.lm.BasicCodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;
import edu.berkeley.cs.nlp.ocular.main.TranscribeOrTrainFont.EMIterationEvaluator;
import edu.berkeley.cs.nlp.ocular.model.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.util.StringHelper;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import indexer.Indexer;
import threading.BetterThreader;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class FontTrainEM {
	
	private boolean accumulateBatchesWithinIter;
	private int minDocBatchSize;
	private int updateDocBatchSize;
	
	private boolean markovVerticalOffset;
	
	private int paddingMinWidth;
	private int paddingMaxWidth;
	
	private int beamSize;
	private int numDecodeThreads;
	private int numMstepThreads;
	private int decodeBatchSize;
	
	public FontTrainEM(boolean accumulateBatchesWithinIter, int minDocBatchSize, int updateDocBatchSize,
			boolean markovVerticalOffset, int paddingMinWidth, int paddingMaxWidth, int beamSize, int numDecodeThreads,
			int numMstepThreads, int decodeBatchSize) {
		this.accumulateBatchesWithinIter = accumulateBatchesWithinIter;
		this.minDocBatchSize = minDocBatchSize;
		this.updateDocBatchSize = updateDocBatchSize;
		this.markovVerticalOffset = markovVerticalOffset;
		this.paddingMinWidth = paddingMinWidth;
		this.paddingMaxWidth = paddingMaxWidth;
		this.beamSize = beamSize;
		this.numDecodeThreads = numDecodeThreads;
		this.numMstepThreads = numMstepThreads;
		this.decodeBatchSize = decodeBatchSize;
	}

	public void run(boolean learnFont,
					int numEMIters,
					List<Document> documents,
					CodeSwitchLanguageModel lm,
					SparseTransitionModel forwardTransitionModel,
					boolean retrainLM,
					Map<String, CharacterTemplate> font,
					Indexer<String> charIndexer,
					EmissionCacheInnerLoop emissionInnerLoop,
					EMIterationEvaluator emIterationEvaluator
					) {
			
		final CharacterTemplate[] templates = loadTemplates(font, charIndexer);
		int numUsableDocs = documents.size();

		if (!learnFont) numEMIters = 0;
		else if (numEMIters <= 0) new RuntimeException("If learnFont=true, then numEMIters must be a positive number.");

		long overallEmissionCacheNanoTime = 0;
		
		for (int iter = 1; (/* learnFont && */ iter <= numEMIters) || (/* !learnFont && */ iter == 1); ++iter) {
			if (learnFont) System.out.println("Training iteration: " + iter + "  (learnFont=true).");
			else System.out.println("Transcribing (learnFont = false).");

			DenseBigramTransitionModel backwardTransitionModel = new DenseBigramTransitionModel(lm);

			// The number of characters assigned to a particular language (to re-estimate language probabilities).
			clearTemplates(templates);

			for (int docNum = 0; docNum < numUsableDocs; ++docNum) {
				Document doc = documents.get(docNum);
				if (learnFont) System.out.println("Training iteration "+iter+" of "+numEMIters+", document: "+(docNum+1)+" of "+numUsableDocs+":  "+doc.baseName());
				else System.out.println("Transcribing document: "+docNum+" of "+numUsableDocs+":  "+doc.baseName());
				doc.loadLineText();

				// e-step

				final PixelType[][][] pixels = doc.loadLineImages();
				TransitionState[][] decodeStates = new TransitionState[pixels.length][0];
				int[][] decodeWidths = new int[pixels.length][0];
				int numBatches = (int) Math.ceil(pixels.length / (double) decodeBatchSize);

				for (int b = 0; b < numBatches; ++b) {
					System.gc();
					System.gc();
					System.gc();

					System.out.println("Batch: " + b);

					int startLine = b * decodeBatchSize;
					int endLine = Math.min((b + 1) * decodeBatchSize, pixels.length);
					PixelType[][][] batchPixels = new PixelType[endLine - startLine][][];
					for (int line = startLine; line < endLine; ++line) {
						batchPixels[line - startLine] = pixels[line];
					}

					final EmissionModel emissionModel = (markovVerticalOffset ? 
							new CachingEmissionModelExplicitOffset(templates, charIndexer, batchPixels, paddingMinWidth, paddingMaxWidth, emissionInnerLoop) : 
							new CachingEmissionModel(templates, charIndexer, batchPixels, paddingMinWidth, paddingMaxWidth, emissionInnerLoop));
					long emissionCacheNanoTime = System.nanoTime();
					emissionModel.rebuildCache();
					overallEmissionCacheNanoTime += (System.nanoTime() - emissionCacheNanoTime);

					long nanoTime = System.nanoTime();
					BeamingSemiMarkovDP dp = new BeamingSemiMarkovDP(emissionModel, forwardTransitionModel, backwardTransitionModel);
					Tuple2<Tuple2<TransitionState[][], int[][]>, Double> decodeStatesAndWidthsAndJointLogProb = dp.decode(beamSize, numDecodeThreads);
					final TransitionState[][] batchDecodeStates = decodeStatesAndWidthsAndJointLogProb._1._1;
					final int[][] batchDecodeWidths = decodeStatesAndWidthsAndJointLogProb._1._2;
					System.out.println("Decode: " + (System.nanoTime() - nanoTime) / 1000000 + "ms");
					
					for (int line = 0; line < emissionModel.numSequences(); ++line) {
						decodeStates[startLine + line] = batchDecodeStates[line];
						decodeWidths[startLine + line] = batchDecodeWidths[line];
					}

					if (learnFont) {
						incrementCounts(emissionModel, batchDecodeStates, batchDecodeWidths);
					}
				}

				// evaluate
				emIterationEvaluator.evaluate(iter, doc, decodeStates);

				// m-step
				{
					int trueMinBatchSize = Math.min(minDocBatchSize, updateDocBatchSize); // min batch size may not exceed standard batch size
					if (docNum+1 == numUsableDocs) { // last document of the set
						if (learnFont) updateFontParameters(iter, templates);
						if (retrainLM) lm = updateLmParameters(lm, decodeStates);
					}
					else if (numUsableDocs - (docNum+1) < trueMinBatchSize) { // next batch will be too small, so lump the remaining documents in with this one
						// no update
					} 
					else if ((docNum+1) % updateDocBatchSize == 0) { // batch is complete
						if (learnFont) updateFontParameters(iter, templates);
						if (retrainLM) lm = updateLmParameters(lm, decodeStates);
					}
				}
			}

		}
		
		System.out.println("Emission cache time: " + overallEmissionCacheNanoTime / 1e9 + "s");
	}
	
	private CharacterTemplate[] loadTemplates(Map<String, CharacterTemplate> font, Indexer<String> charIndexer) {
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

	private void incrementCounts(final EmissionModel emissionModel, final TransitionState[][] batchDecodeStates, final int[][] batchDecodeWidths) {
		long nanoTime;
		nanoTime = System.nanoTime();
		BetterThreader.Function<Integer, Object> func = new BetterThreader.Function<Integer, Object>() {
			public void call(Integer line, Object ignore) {
				emissionModel.incrementCounts(line, batchDecodeStates[line], batchDecodeWidths[line]);
			}
		};
		BetterThreader<Integer, Object> threader = new BetterThreader<Integer, Object>(func, numMstepThreads);
		for (int line = 0; line < emissionModel.numSequences(); ++line)
			threader.addFunctionArgument(line);
		threader.run();
		System.out.println("Increment counts: " + (System.nanoTime() - nanoTime) / 1000000 + "ms");
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
	 * Hard-EM update on language probabilities
	 */
	private CodeSwitchLanguageModel updateLmParameters(CodeSwitchLanguageModel lm, TransitionState[][] decodeStates) {
		long nanoTime = System.nanoTime();
		Indexer<String> langIndexer = lm.getLanguageIndexer();
		
		//
		// Initialize containers for counts
		//
		int numLanguages = langIndexer.size();
		int[] languageCounts = new int[numLanguages];
		Arrays.fill(languageCounts, 1); // one-count smooth
		
		//
		// Pass over the decoded states to accumulate counts
		//
		for (int line = 0; line < decodeStates.length; ++line) {
			if (decodeStates[line] != null) {
				for (int i = 0; i < decodeStates[line].length; ++i) {
					int currLanguage = decodeStates[line][i].getLanguageIndex();
					if (currLanguage >= 0) 
						languageCounts[currLanguage] += 1;
				}
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

}
