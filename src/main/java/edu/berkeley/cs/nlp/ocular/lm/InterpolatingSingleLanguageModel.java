package edu.berkeley.cs.nlp.ocular.lm;

import java.util.List;
import java.util.Set;

import edu.berkeley.cs.nlp.ocular.util.ArrayHelper;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;
import tberg.murphy.indexer.Indexer;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class InterpolatingSingleLanguageModel implements SingleLanguageModel {
	private static final long serialVersionUID = 1L;

	private SingleLanguageModel[] subModels;
	private double[] interpWeights;
	private int numModels;

	private Indexer<String> charIndexer = null;
	private Set<Integer> activeCharacters = null;
	private int maxOrder = -1;

	public InterpolatingSingleLanguageModel(List<Tuple2<SingleLanguageModel, Double>> subModelsAndinterpWeights) {
		numModels = subModelsAndinterpWeights.size();
		
		subModels = new SingleLanguageModel[numModels];
		interpWeights = new double[numModels];
		
		double totalInterpWeight = 0.0;
		for (int i = 0; i < numModels; ++i) {
			Tuple2<SingleLanguageModel, Double> modelAndWeight = subModelsAndinterpWeights.get(i);
			subModels[i] = modelAndWeight._1;
			interpWeights[i] = modelAndWeight._2;
			totalInterpWeight += interpWeights[i];
			
			if (charIndexer == null) {
				charIndexer = subModels[i].getCharacterIndexer();
				activeCharacters = subModels[i].getActiveCharacters();
				int thisMaxOrder = subModels[i].getMaxOrder();
				if (thisMaxOrder > maxOrder)
					maxOrder = thisMaxOrder;
			} else if (charIndexer != subModels[i].getCharacterIndexer()) {
				throw new RuntimeException("Sub-models don't all use the same character indexer");
			} else if (activeCharacters != subModels[i].getActiveCharacters()) {
				throw new RuntimeException("Sub-models don't all use the same active-character set");
			}
		}
		for (int i = 0; i < numModels; ++i) {
			interpWeights[i] /= totalInterpWeight;
		}
	}

	@Override
	public double getCharNgramProb(int[] context, int c) {
		double probSum = 0.0;
		for (int i = 0; i < numModels; ++i) {
			int[] shrunkenContext = subModels[i].shrinkContext(context); // context may be different for different submodels
			probSum += subModels[i].getCharNgramProb(shrunkenContext, c) * interpWeights[i];
		}
		return probSum;
	}

	@Override
	public Indexer<String> getCharacterIndexer() {
		return charIndexer;
	}

	@Override
	public Set<Integer> getActiveCharacters() {
		return activeCharacters;
	}
	
	@Override
	public int getMaxOrder() {
		return maxOrder;
	}

	@Override
	public int[] shrinkContext(int[] originalContext) {
		int[] newContext = originalContext;
		while (!containsContext(newContext) && newContext.length > 0) {
			newContext = ArrayHelper.takeRight(newContext, newContext.length - 1);
		}
		return newContext;
	}
	
	@Override
	public boolean containsContext(int[] context) {
		for (SingleLanguageModel slm : subModels) {
			if (slm.containsContext(context)) {
				return true;
			}
		}
		return false;
	}
	
	public SingleLanguageModel getSubModel(int i) {
		return subModels[i];
	}

}
