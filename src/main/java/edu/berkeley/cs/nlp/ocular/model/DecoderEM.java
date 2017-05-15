package edu.berkeley.cs.nlp.ocular.model;

import static edu.berkeley.cs.nlp.ocular.util.Tuple2.Tuple2;

import java.text.SimpleDateFormat;
import java.util.Calendar;

import edu.berkeley.cs.nlp.ocular.data.Document;
import edu.berkeley.cs.nlp.ocular.gsm.GlyphSubstitutionModel;
import edu.berkeley.cs.nlp.ocular.image.ImageUtils.PixelType;
import edu.berkeley.cs.nlp.ocular.lm.CodeSwitchLanguageModel;
import edu.berkeley.cs.nlp.ocular.lm.SingleLanguageModel;
import edu.berkeley.cs.nlp.ocular.model.em.BeamingSemiMarkovDP;
import edu.berkeley.cs.nlp.ocular.model.em.DenseBigramTransitionModel;
import edu.berkeley.cs.nlp.ocular.model.emission.EmissionModel;
import edu.berkeley.cs.nlp.ocular.model.emission.EmissionModel.EmissionModelFactory;
import edu.berkeley.cs.nlp.ocular.model.transition.CharacterNgramTransitionModel;
import edu.berkeley.cs.nlp.ocular.model.transition.CharacterNgramTransitionModelMarkovOffset;
import edu.berkeley.cs.nlp.ocular.model.transition.CodeSwitchTransitionModel;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel;
import edu.berkeley.cs.nlp.ocular.model.transition.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import tberg.murphy.threading.BetterThreader;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class DecoderEM {

	private EmissionModelFactory emissionModelFactory;

	private boolean allowGlyphSubstitution;
	private double noCharSubPrior;
	private boolean elideAnything;
	private boolean allowLanguageSwitchOnPunct;
	private boolean markovVerticalOffset;
	
	private int beamSize;
	private int numDecodeThreads;
	private int numMstepThreads;
	private int decodeBatchSize;
	
	public DecoderEM(EmissionModelFactory emissionModelFactory, boolean allowGlyphSubstitution, double noCharSubPrior, boolean elideAnything,
			boolean allowLanguageSwitchOnPunct, boolean markovVerticalOffset,
			int beamSize, int numDecodeThreads, int numMstepThreads, int decodeBatchSize) {
		this.emissionModelFactory = emissionModelFactory;
		this.allowGlyphSubstitution = allowGlyphSubstitution;
		this.noCharSubPrior = noCharSubPrior;
		this.elideAnything = elideAnything;
		this.allowLanguageSwitchOnPunct = allowLanguageSwitchOnPunct;
		this.markovVerticalOffset = markovVerticalOffset;
		this.beamSize = beamSize;
		this.numDecodeThreads = numDecodeThreads;
		this.numMstepThreads = numMstepThreads;
		this.decodeBatchSize = decodeBatchSize;
	}

	public Tuple2<DecodeState[][], Double> computeEStep(
			Document doc, boolean updateFontParameterCounts,
			CodeSwitchLanguageModel lm, GlyphSubstitutionModel gsm, final CharacterTemplate[] templates,
			DenseBigramTransitionModel backwardTransitionModel) {
		
		final PixelType[][][] pixels = doc.loadLineImages();
		
		DecodeState[][] allDecodeStates = new DecodeState[pixels.length][0];

		long totalDecodeNanoTime = 0;
		long totalEmitNanoTime = 0;
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

			System.out.println("Initializing EmissionModel    " + (new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime())));
			final EmissionModel batchEmissionModel = emissionModelFactory.make(templates, batchPixels);
			System.out.println("Rebuilding cache    " + (new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime())));
			//long emissionCacheNanoTime = System.nanoTime();
			long nanoTime = System.nanoTime();
			batchEmissionModel.rebuildCache();
			totalEmitNanoTime += (System.nanoTime() - nanoTime);
			System.out.println("Done rebuilding cache    " + (new SimpleDateFormat("yyyy/MM/dd HH:mm:ss").format(Calendar.getInstance().getTime())));
			//overallEmissionCacheNanoTime += (System.nanoTime() - emissionCacheNanoTime);

			nanoTime = System.nanoTime();
			System.out.println("Constructing forwardTransitionModel");
			SparseTransitionModel forwardTransitionModel = constructTransitionModel(lm, gsm);
			BeamingSemiMarkovDP dp = new BeamingSemiMarkovDP(batchEmissionModel, forwardTransitionModel, backwardTransitionModel);
			System.out.println("Ready to run decoder");
			Tuple2<Tuple2<TransitionState[][], int[][]>, Double> decodeStatesAndWidthsAndJointLogProb = dp.decode(beamSize, numDecodeThreads);
			System.out.println("Done running decoder");
			totalDecodeNanoTime += (System.nanoTime() - nanoTime);
			final TransitionState[][] batchDecodeStates = decodeStatesAndWidthsAndJointLogProb._1._1;
			final int[][] batchDecodeWidths = decodeStatesAndWidthsAndJointLogProb._1._2;
			totalJointLogProb += decodeStatesAndWidthsAndJointLogProb._2;
			for (int batchLine = 0; batchLine < batchEmissionModel.numSequences(); ++batchLine) {
				int line = startLine + batchLine;
				TransitionState[] decodeStates = batchDecodeStates[batchLine];
				int[] decodeWidths = batchDecodeWidths[batchLine];
				allDecodeStates[line] = new DecodeState[decodeStates.length];
				int stateStartCol = 0;
				for (int di=0; di<decodeStates.length; ++di) {
					int charAndPadWidth = decodeWidths[di];
					int padWidth = batchEmissionModel.getPadWidth(batchLine, stateStartCol, decodeStates[di], charAndPadWidth);
					int exposure = batchEmissionModel.getExposure(batchLine, stateStartCol, decodeStates[di], charAndPadWidth);
					int verticalOffset = batchEmissionModel.getOffset(batchLine, stateStartCol, decodeStates[di], charAndPadWidth);
					allDecodeStates[line][di] = new DecodeState(decodeStates[di], charAndPadWidth, padWidth, exposure, verticalOffset);
					stateStartCol += charAndPadWidth;
				}
			}

			if (updateFontParameterCounts) {
				System.out.println("Ready to run increment counts");
				incrementCounts(batchEmissionModel, batchDecodeStates, batchDecodeWidths);
			}
		}
		System.out.println("Emission cache: " + (totalEmitNanoTime / 1000000) + "ms");
		System.out.println("Decode: " + (totalDecodeNanoTime / 1000000) + "ms");
		double avgLogProb = totalJointLogProb / numBatches;
		return Tuple2(allDecodeStates, avgLogProb);
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
				transitionModel = new CodeSwitchTransitionModel(codeSwitchLM, allowLanguageSwitchOnPunct, codeSwitchGSM, allowGlyphSubstitution, noCharSubPrior, elideAnything);
				System.out.println("Using CodeSwitchLanguageModel, GlyphSubstitutionModel, and CodeSwitchTransitionModel");
			}
		}
		else { // only one language, default to original (monolingual) Ocular code because it will be faster.
			SingleLanguageModel singleLm = codeSwitchLM.get(0);
			if (markovVerticalOffset) {
				transitionModel = new CharacterNgramTransitionModelMarkovOffset(singleLm);
				System.out.println("Using OnlyOneLanguageCodeSwitchLM and CharacterNgramTransitionModelMarkovOffset");
			} else {
				transitionModel = new CharacterNgramTransitionModel(singleLm);
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
