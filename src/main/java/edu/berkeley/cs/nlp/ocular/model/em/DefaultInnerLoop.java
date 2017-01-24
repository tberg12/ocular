package edu.berkeley.cs.nlp.ocular.model.em;

import edu.berkeley.cs.nlp.ocular.model.CharacterTemplate;
import tberg.murphy.gpu.CudaUtil;

/**
 * @author Taylor Berg-Kirkpatrick (tberg@eecs.berkeley.edu)
 */
public class DefaultInnerLoop implements EmissionCacheInnerLoop {
	
	int numThreads;
	float[][] whiteTemplates;
	float[][] blackTemplates;
	int[] templateNumIndices;
	int[] templateIndicesOffsets;
	int maxTemplateWidth;
	int minTemplateWidth;

	public DefaultInnerLoop(int numThreads) {
		this.numThreads = numThreads;
	}
	
	public void startup(float[][] whiteTemplates, float[][] blackTemplates, int[] templateNumIndices, int[] templateIndicesOffsets, int minTemplateWidth, int maxTemplateWidth, int maxSequenceLength, int totalTemplateNumIndices) {
		this.whiteTemplates = whiteTemplates;
		this.blackTemplates = blackTemplates;
		this.templateNumIndices = templateNumIndices;
		this.templateIndicesOffsets = templateIndicesOffsets;
		this.maxTemplateWidth = maxTemplateWidth;
		this.minTemplateWidth = minTemplateWidth;
	}

	public void shutdown() {
	}

	public void compute(final float[] scores, final float[] whiteObservations, final float[] blackObservations, final int sequenceLength) {
		for (int tw=minTemplateWidth; tw<=maxTemplateWidth; ++tw) {
			float[] whiteTemplatesForWidth = whiteTemplates[tw-minTemplateWidth];
			float[] blackTemplateForWidth = blackTemplates[tw-minTemplateWidth];
			for (int t=0; t<(sequenceLength-tw)+1; ++t) {
				for (int i=0; i<templateNumIndices[tw-minTemplateWidth]; ++i) {
					float score = 0.0f;
					for (int j=0; j<tw*CharacterTemplate.LINE_HEIGHT; ++j) {
						score += whiteObservations[t*CharacterTemplate.LINE_HEIGHT+j] * whiteTemplatesForWidth[i*tw*CharacterTemplate.LINE_HEIGHT+j];
					}
					scores[templateIndicesOffsets[tw-minTemplateWidth]*sequenceLength + CudaUtil.flatten(sequenceLength, templateNumIndices[tw-minTemplateWidth], t, i)] += score;
				}
			}
			for (int t=0; t<(sequenceLength-tw)+1; ++t) {
				for (int i=0; i<templateNumIndices[tw-minTemplateWidth]; ++i) {
					float score = 0.0f;
					for (int j=0; j<tw*CharacterTemplate.LINE_HEIGHT; ++j) {
						score += blackObservations[t*CharacterTemplate.LINE_HEIGHT+j] * blackTemplateForWidth[i*tw*CharacterTemplate.LINE_HEIGHT+j];
					}
					scores[templateIndicesOffsets[tw-minTemplateWidth]*sequenceLength + CudaUtil.flatten(sequenceLength, templateNumIndices[tw-minTemplateWidth], t, i)] += score;
				}
			}
		}
	}

	public int numOuterThreads() {
		return numThreads;
	}

	public int numPopulateThreads() {
		return 1;
	}

}
