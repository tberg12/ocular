package edu.berkeley.cs.nlp.ocular.util;

import java.io.Serializable;
import java.util.Comparator;

/**
 * @author Dan Klein
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class Tuple2<A1, A2> implements Serializable {
	static final long serialVersionUID = 52;

	public final A1 _1;
	public final A2 _2;

	public boolean equals(Object o) {
		if (this == o)
			return true;
		if (!(o instanceof Tuple2))
			return false;

		@SuppressWarnings("rawtypes")
		final Tuple2 tuple = (Tuple2) o;

		if (_1 != null ? !_1.equals(tuple._1) : tuple._1 != null)
			return false;
		if (_2 != null ? !_2.equals(tuple._2) : tuple._2 != null)
			return false;

		return true;
	}

	public int hashCode() {
		int result;
		result = (_1 != null ? _1.hashCode() : 0);
		result = 29 * result + (_2 != null ? _2.hashCode() : 0);
		return result;
	}

	public String toString() {
		return "(" + _1 + ", " + _2 + ")";
	}

	public Tuple2(A1 _1, A2 _2) {
		this._1 = _1;
		this._2 = _2;
	}

	public static <A1, A2> Tuple2<A1, A2> Tuple2(A1 _1, A2 _2) {
		return new Tuple2<A1, A2>(_1, _2);
	}

	public static class LexicographicTuple2Comparator<A1, A2> implements Comparator<Tuple2<A1, A2>> {
		Comparator<A1> _1Comparator;
		Comparator<A2> _2Comparator;

		public int compare(Tuple2<A1, A2> tuple1, Tuple2<A1, A2> tuple2) {
			int _1Compare = _1Comparator.compare(tuple1._1, tuple2._1);
			if (_1Compare != 0)
				return _1Compare;
			return _2Comparator.compare(tuple1._2, tuple2._2);
		}

		public LexicographicTuple2Comparator(Comparator<A1> _1Comparator, Comparator<A2> _2Comparator) {
			this._1Comparator = _1Comparator;
			this._2Comparator = _2Comparator;
		}
	}

	public static class DefaultLexicographicTuple2Comparator<A1 extends Comparable<A1>, A2 extends Comparable<A2>>
			implements Comparator<Tuple2<A1, A2>> {

		public int compare(Tuple2<A1, A2> x, Tuple2<A1, A2> y) {
			int _1Compare = x._1.compareTo(y._1);
			if (_1Compare != 0) {
				return _1Compare;
			}
			return y._2.compareTo(y._2);
		}

	}

}
