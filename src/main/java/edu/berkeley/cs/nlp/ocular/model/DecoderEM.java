package edu.berkeley.cs.nlp.ocular.model;

import static edu.berkeley.cs.nlp.ocular.util.Tuple2.makeTuple2;

import edu.berkeley.cs.nlp.ocular.data.ImageLoader.Document;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.sub.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import indexer.Indexer;
import threading.BetterThreader;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhg@cs.utexas.edu)
 */
public class DecoderEM {

	private EmissionCacheInnerLoop emissionInnerLoop;

	private boolean allowGlyphSubstitution;
	private double noCharSubPrior;
	private boolean allowLanguageSwitchOnPunct;
	private boolean markovVerticalOffset;
	
	private int paddingMinWidth;
	private int paddingMaxWidth;
	
	private int beamSize;
	private int numDecodeThreads;
	private int numMstepThreads;
	private int decodeBatchSize;
	
	private Indexer<String> charIndexer;

	public DecoderEM(EmissionCacheInnerLoop emissionInnerLoop, boolean allowGlyphSubstitution, double noCharSubPrior,
			boolean allowLanguageSwitchOnPunct, boolean markovVerticalOffset, int paddingMinWidth, int paddingMaxWidth,
			int beamSize, int numDecodeThreads, int numMstepThreads, int decodeBatchSize, Indexer<String> charIndexer) {
		this.emissionInnerLoop = emissionInnerLoop;
		this.allowGlyphSubstitution = allowGlyphSubstitution;
		this.noCharSubPrior = noCharSubPrior;
		this.allowLanguageSwitchOnPunct = allowLanguageSwitchOnPunct;
		this.markovVerticalOffset = markovVerticalOffset;
		this.paddingMinWidth = paddingMinWidth;
		this.paddingMaxWidth = paddingMaxWidth;
		this.beamSize = beamSize;
		this.numDecodeThreads = numDecodeThreads;
		this.numMstepThreads = numMstepThreads;
		this.decodeBatchSize = decodeBatchSize;
		this.charIndexer = charIndexer;
	}

	public Tuple2<Tuple2<TransitionState[][], int[][]>, Double> computeEStep(
			Document doc, boolean learnFont,
			CodeSwitchLanguageModel lm, GlyphSubstitutionModel gsm, final CharacterTemplate[] templates,
			DenseBigramTransitionModel backwardTransitionModel) {
		final PixelType[][][] pixels = doc.loadLineImages();
		TransitionState[][] decodeStates = new TransitionState[pixels.length][0];
		int[][] decodeWidths = new int[pixels.length][0];

		int totalNanoTime = 0;
		double totalJointLogProb = 0.0;
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
			//long emissionCacheNanoTime = System.nanoTime();
			emissionModel.rebuildCache();
			//overallEmissionCacheNanoTime += (System.nanoTime() - emissionCacheNanoTime);

			long nanoTime = System.nanoTime();
			SparseTransitionModel forwardTransitionModel = constructTransitionModel(lm, gsm);
			BeamingSemiMarkovDP dp = new BeamingSemiMarkovDP(emissionModel, forwardTransitionModel, backwardTransitionModel);
			Tuple2<Tuple2<TransitionState[][], int[][]>, Double> decodeStatesAndWidthsAndJointLogProb = dp.decode(beamSize, numDecodeThreads);
			totalNanoTime += (System.nanoTime() - nanoTime);
			final TransitionState[][] batchDecodeStates = decodeStatesAndWidthsAndJointLogProb._1._1;
			final int[][] batchDecodeWidths = decodeStatesAndWidthsAndJointLogProb._1._2;
			totalJointLogProb += decodeStatesAndWidthsAndJointLogProb._2;
			for (int line = 0; line < emissionModel.numSequences(); ++line) {
				decodeStates[startLine + line] = batchDecodeStates[line];
				decodeWidths[startLine + line] = batchDecodeWidths[line];
			}

			if (learnFont) {
				incrementCounts(emissionModel, batchDecodeStates, batchDecodeWidths);
			}
			
		}
		System.out.println("Decode: " + (totalNanoTime / 1000000) + "ms");
		double avgLogProb = totalJointLogProb / numBatches;
		return makeTuple2(makeTuple2(decodeStates, decodeWidths), avgLogProb);
	}

	private SparseTransitionModel constructTransitionModel(CodeSwitchLanguageModel codeSwitchLM, GlyphSubstitutionModel codeSwitchGSM) {
		SparseTransitionModel transitionModel;
		
		boolean multilingual = codeSwitchLM.getLanguageIndexer().size() > 1;
		if (multilingual || allowGlyphSubstitution) {
			if (markovVerticalOffset) {
				if (allowGlyphSubstitution)
					throw new RuntimeException("Markov vertical offset transition model not currently supported with glyph substitution.");
				else
					throw new RuntimeException("Markov vertical offset transition model not currently supported for multiple languages.");
			}
			else { 
				transitionModel = new CodeSwitchTransitionModel(codeSwitchLM, allowLanguageSwitchOnPunct, codeSwitchGSM, allowGlyphSubstitution, noCharSubPrior);
				System.out.println("Using CodeSwitchLanguageModel, GlyphSubstitutionModel, and CodeSwitchTransitionModel");
			}
		}
		else { // only one language, default to original (monolingual) Ocular code because it will be faster.
			SingleLanguageModel singleLm = codeSwitchLM.get(0);
			if (markovVerticalOffset) {
				transitionModel = new CharacterNgramTransitionModelMarkovOffset(singleLm, singleLm.getMaxOrder());
				System.out.println("Using OnlyOneLanguageCodeSwitchLM and CharacterNgramTransitionModelMarkovOffset");
			} else {
				transitionModel = new CharacterNgramTransitionModel(singleLm, singleLm.getMaxOrder());
				System.out.println("Using OnlyOneLanguageCodeSwitchLM and CharacterNgramTransitionModel");
			}
		}
		
		return transitionModel;
	}
	
	private void incrementCounts(final EmissionModel emissionModel, final TransitionState[][] batchDecodeStates, final int[][] batchDecodeWidths) {
		long nanoTime = System.nanoTime();
		BetterThreader.Function<Integer, Object> func = new BetterThreader.Function<Integer, Object>() {
			public void call(Integer line, Object ignore) {
				emissionModel.incrementCounts(line, batchDecodeStates[line], batchDecodeWidths[line]);
			}
		};
		BetterThreader<Integer, Object> threader = new BetterThreader<Integer, Object>(func, numMstepThreads);
		for (int line = 0; line < emissionModel.numSequences(); ++line)
			threader.addFunctionArgument(line);
		threader.run();
		System.out.println("Increment counts: " + ((System.nanoTime() - nanoTime) / 1000000) + "ms");
	}

}
