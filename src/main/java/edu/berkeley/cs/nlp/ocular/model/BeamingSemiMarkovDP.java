package edu.berkeley.cs.nlp.ocular.model;

import static edu.berkeley.cs.nlp.ocular.util.Tuple2.makeTuple2;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import threading.BetterThreader;
import util.GeneralPriorityQueue;
import arrays.a;
import edu.berkeley.cs.nlp.ocular.model.SparseTransitionModel.TransitionState;
import edu.berkeley.cs.nlp.ocular.util.Tuple2;

public class BeamingSemiMarkovDP {
	
	private static class BeamState {
		private final TransitionState transState;
		public double score = Double.NEGATIVE_INFINITY;
		public Tuple2<Integer,TransitionState> backPointer = null;
		public BeamState(TransitionState transState) {
			this.transState = transState;
		}
		public int hashCode() {
			return transState.hashCode();
		}
		public boolean equals(Object obj) {
			if (obj instanceof BeamState) {
				return transState.equals(((BeamState) obj).transState);
			} else {
				return false;
			}
		}
	}
	
	private GeneralPriorityQueue<BeamState>[][] alphas;
	double[][][] betas;
	private SparseTransitionModel forwardTransitionModel;
	private DenseBigramTransitionModel backwardTransitionModel;
	private EmissionModel emissionModel;

	public BeamingSemiMarkovDP(EmissionModel emissionModel, SparseTransitionModel forwardTransitionModel, DenseBigramTransitionModel backwardTransitionModel) {
		this.emissionModel = emissionModel;
		this.forwardTransitionModel = forwardTransitionModel;
		this.backwardTransitionModel = backwardTransitionModel;
		this.alphas = new GeneralPriorityQueue[emissionModel.numSequences()][];
		for (int d=0; d<emissionModel.numSequences(); ++d) {
			this.alphas[d] = new GeneralPriorityQueue[emissionModel.sequenceLength(d)+1];
			for (int t=0; t<emissionModel.sequenceLength(d)+1; ++t) {
				this.alphas[d][t] = new GeneralPriorityQueue<BeamState>();
			}
		}
		this.betas = new double[emissionModel.numSequences()][][];
		for (int d=0; d<emissionModel.numSequences(); ++d) {
			this.betas[d] = new double[emissionModel.sequenceLength(d)+1][emissionModel.getCharIndexer().size()];
		}
	}

	public Tuple2<Tuple2<TransitionState[][],int[][]>,Double> decode(int beamSize) {
		TransitionState[][] decodeStates = new TransitionState[emissionModel.numSequences()][];
		int[][] decodeWidths = new int[emissionModel.numSequences()][];
		
		Collection<BeamState> startStates = null;
		double logJointProb = Double.NEGATIVE_INFINITY;
		for (int d=0; d<emissionModel.numSequences(); ++d) {
			Tuple2<Double,Collection<BeamState>> logJointProbAndNextStartStates = doForwardPassLogSpace(d, beamSize, startStates);
			logJointProb = logJointProbAndNextStartStates._1;
			startStates = logJointProbAndNextStartStates._2;
		}
		TransitionState finalState = null;
		for (int d=emissionModel.numSequences()-1; d>=0; --d) {
			Tuple2<Tuple2<TransitionState[],int[]>,TransitionState> statesAndWidthsAndNextFinalState = followBackpointers(d, finalState);
			decodeStates[d] = statesAndWidthsAndNextFinalState._1._1;
			decodeWidths[d] = statesAndWidthsAndNextFinalState._1._2;
			finalState = statesAndWidthsAndNextFinalState._2;
		}
		return makeTuple2(makeTuple2(decodeStates, decodeWidths), logJointProb);
	}
	
	public Tuple2<Tuple2<TransitionState[][],int[][]>,Double> decode(final int beamSize, int numThreads) {
		System.out.print("Decoding");
		
		if (numThreads == 1) return decode(beamSize);
		
		final TransitionState[][] decodeStates = new TransitionState[emissionModel.numSequences()][];
		final int[][] decodeWidths = new int[emissionModel.numSequences()][];
		final int blockSize = (int) Math.ceil(((double) emissionModel.numSequences()) / ((double)numThreads));
		final double[] logJointProb = new double[] {0.0};
		{
			BetterThreader.Function<Integer,Object> func = new BetterThreader.Function<Integer,Object>(){public void call(Integer b, Object ignore){
				double blockLogJointProb = Double.NEGATIVE_INFINITY;
				Collection<BeamState> startStates = null;
				for (int d=b*blockSize; d<(b+1)*blockSize; ++d) {
					if (d < emissionModel.numSequences()) {
						Tuple2<Double,Collection<BeamState>> logJointProbAndNextStartStates = doForwardPassLogSpace(d, beamSize, startStates);
						blockLogJointProb = logJointProbAndNextStartStates._1;
						startStates = logJointProbAndNextStartStates._2;
					}
				}
				TransitionState finalState = null;
				for (int d=(b+1)*blockSize-1; d>=b*blockSize; --d) {
					if (d < emissionModel.numSequences()) {
						Tuple2<Tuple2<TransitionState[],int[]>,TransitionState> statesAndWidthsAndNextFinalState = followBackpointers(d, finalState);
						decodeStates[d] = statesAndWidthsAndNextFinalState._1._1;
						decodeWidths[d] = statesAndWidthsAndNextFinalState._1._2;
						finalState = statesAndWidthsAndNextFinalState._2;
					}
				}
				synchronized (logJointProb) {
					logJointProb[0] += blockLogJointProb;
				}
			}};
			BetterThreader<Integer,Object> threader = new BetterThreader<Integer,Object>(func, numThreads);
			for (int b=0; b<numThreads; ++b) threader.addFunctionArgument(b);
			threader.run();
		}
		
		System.out.println();
		return makeTuple2(makeTuple2(decodeStates, decodeWidths), logJointProb[0]);
	}
	
	private Tuple2<Double,Collection<BeamState>> doForwardPassLogSpace(int d, int beamSize, Collection<BeamState> startStates) {
		System.out.print(".");
		
//		System.out.printf("Backward pass: %d%n", d);
		doDenseCoarseBackwardPassLogSpace(d, betas[d]);
		
//		System.out.printf("Forward pass: %d%n", d);
		for (GeneralPriorityQueue<BeamState> queue : alphas[d]) queue.clear();
		for (int t=0; t<emissionModel.sequenceLength(d)+1; ++t) {
			if (t == 0) {
				for (BeamState startBeamState : (startStates == null || startStates.isEmpty() ? addNullBackpointers(forwardTransitionModel.startStates(d)) : startStates)) {
					TransitionState nextTs = startBeamState.transState;
					double startLogProb = startBeamState.score;
					if (startLogProb != Double.NEGATIVE_INFINITY) {
						int c = nextTs.getCharIndex(); 
						for (int w : emissionModel.allowedWidths(c)) {
							if (t + w < emissionModel.sequenceLength(d)+1) {
								int nextT = t + w;
								double emissionLogProb = emissionModel.logProb(d, t, nextTs, nextT-t);
								double score = startLogProb + emissionLogProb;
								if (score != Double.NEGATIVE_INFINITY) {
									addToBeam(alphas[d][nextT], nextTs, score, betas[d][nextT][nextTs.getCharIndex()], new Tuple2<Integer,TransitionState>(0, startBeamState.backPointer._2), beamSize);
								}
							}
						}
					}
				}
			} else {
				for (BeamState beamState : alphas[d][t].getObjects()) {
					Collection<Tuple2<TransitionState,Double>> allowedTrans = beamState.transState.forwardTransitions();
					for (Tuple2<TransitionState,Double> trans : allowedTrans) {
						TransitionState nextTs = trans._1;
						double transLogProb = trans._2;
						int c =nextTs.getCharIndex();
						for (int w : emissionModel.allowedWidths(c)) {
							if (t + w < emissionModel.sequenceLength(d)+1) {
								int nextT = t + w;
								double emissionLogProb = emissionModel.logProb(d, t, nextTs, nextT-t);
								double score = beamState.score + transLogProb + emissionLogProb;
								if (score != Double.NEGATIVE_INFINITY) {
									addToBeam(alphas[d][nextT], nextTs, score, betas[d][nextT][nextTs.getCharIndex()], makeTuple2(t, beamState.transState), beamSize);
								}
							}
						}
					}
				}
			}
		}
		
		double bestFinalScore = Double.NEGATIVE_INFINITY;
		Map<TransitionState,BeamState> wrappedStartStatesMap = new HashMap<TransitionState,BeamState>();
		for (BeamState endBeamState : alphas[d][emissionModel.sequenceLength(d)].getObjects()) {
			double endScore = endBeamState.score + endBeamState.transState.endLogProb();
			if (endScore != Double.NEGATIVE_INFINITY) {
				if (endScore > bestFinalScore) {
					bestFinalScore = endScore;
				}
				for (Tuple2<TransitionState,Double> startTransitionPair : endBeamState.transState.nextLineStartStates()) {
					double score = endScore + startTransitionPair._2;
					if (score != Double.NEGATIVE_INFINITY) {
						BeamState startBeamState = wrappedStartStatesMap.get(startTransitionPair._1);
						if (startBeamState == null) {
							startBeamState = new BeamState(startTransitionPair._1);
							startBeamState.score = Double.NEGATIVE_INFINITY;
							startBeamState.backPointer = new Tuple2<Integer, TransitionState>(-1, null);
							wrappedStartStatesMap.put(startTransitionPair._1, startBeamState);
						}
						if (score > startBeamState.score) {
							startBeamState.score = score;
							startBeamState.backPointer = makeTuple2(-1, endBeamState.transState);
						}
					}
				}
			}
		}
		Collection<BeamState> wrappedStartStates = new ArrayList<BeamState>();
		for (Map.Entry<TransitionState, BeamState> entry : wrappedStartStatesMap.entrySet()) {
			wrappedStartStates.add(entry.getValue());
		}
		
		return makeTuple2(bestFinalScore, wrappedStartStates);
	}
	
	private static void addToBeam(GeneralPriorityQueue<BeamState> queue, TransitionState nextTs, double score, double forwardScore, Tuple2<Integer,TransitionState> backPointer, int beamSize) {
		double priority = -(score+forwardScore);
		if (queue.isEmpty() || priority < queue.getPriority()) {
			BeamState key = new BeamState(nextTs);
			if (queue.containsKey(key)) {
				queue.decreasePriority(key, priority);
			} else {
				queue.setPriority(key, priority);
			}
			BeamState object = queue.getObject(key);
			if (object.score < score) {
				object.score = score;
				object.backPointer = backPointer;
			}
			while (queue.size() > beamSize) {
				queue.removeFirst(); 
			}
		}
	}
	
	private static Collection<BeamState> addNullBackpointers(Collection<Tuple2<TransitionState,Double>> without) {
		List<BeamState> with = new ArrayList<BeamState>();
		for (Tuple2<TransitionState,Double> startPair : without) {
			BeamState beamState = new BeamState(startPair._1);
			beamState.score = startPair._2;
			beamState.backPointer = makeTuple2(-1, null);
			with.add(beamState);
		}
		return with;
	}

	private Tuple2<Tuple2<TransitionState[],int[]>,TransitionState> followBackpointers(int d, TransitionState finalTs) {
		List<TransitionState> transStateDecodeList = new ArrayList<TransitionState>();
		List<Integer> widthsDecodeList = new ArrayList<Integer>();
		TransitionState bestFinalTs = null;
		if (finalTs == null) {
			double bestFinalScore = Double.NEGATIVE_INFINITY;
			for (BeamState beamState : alphas[d][emissionModel.sequenceLength(d)].getObjects()) {
				double score = beamState.score + beamState.transState.endLogProb();
				if (score > bestFinalScore) {
					bestFinalScore = score;
					bestFinalTs = beamState.transState;
				}
			}
		} else {
			bestFinalTs = finalTs;
		}

		int currentT = emissionModel.sequenceLength(d);
		TransitionState nextFinalTs = null;
		TransitionState currentTs = bestFinalTs;
		while (true) {
			Tuple2<Integer,TransitionState> backpointer = alphas[d][currentT].getObject(new BeamState(currentTs)).backPointer;
			int width =  currentT - backpointer._1;
			transStateDecodeList.add(currentTs);
			widthsDecodeList.add(width);
			currentT = backpointer._1;
			currentTs = backpointer._2;
			if (currentT == 0) {
				nextFinalTs = currentTs;
				break;
			}
		}

		Collections.reverse(transStateDecodeList);
		Collections.reverse(widthsDecodeList);
		int[] widthsDecode = a.toIntArray(widthsDecodeList);
		return makeTuple2(makeTuple2(transStateDecodeList.toArray(new TransitionState[0]), widthsDecode), nextFinalTs);
	}
	
	private void doDenseCoarseBackwardPassLogSpace(int d, double[][] betas) {
		for (int t=emissionModel.sequenceLength(d); t>=0; --t) {
			Arrays.fill(betas[t], Double.NEGATIVE_INFINITY);
			if (t==emissionModel.sequenceLength(d)) {
				for (int c=0; c<emissionModel.getCharIndexer().size(); ++c) {
					betas[t][c] = backwardTransitionModel.endLogProb(c);
				}
			} else {
				for (int nextC=0; nextC<emissionModel.getCharIndexer().size(); ++nextC) {
					double betaWithoutTrans = Double.NEGATIVE_INFINITY;
					int[] allowedWidths = emissionModel.allowedWidths(nextC);
					for (int w : allowedWidths) {
						if (t + w <= emissionModel.sequenceLength(d)) {
							double emissionLogProb = emissionModel.logProb(d, t, nextC, w);
							betaWithoutTrans = Math.max(betaWithoutTrans, emissionLogProb + betas[t+w][nextC]);
						}
					}
					double[] betasCol = betas[t];
					double[] logTransProbs = backwardTransitionModel.backwardTransitions(nextC);
					for (int c=0; c<emissionModel.getCharIndexer().size(); ++c) {
						betasCol[c] = Math.max(betasCol[c], logTransProbs[c] + betaWithoutTrans);
					}
				}
			}
		}
	}
	
}
