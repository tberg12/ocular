package edu.berkeley.cs.nlp.ocular.util;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import edu.berkeley.cs.nlp.ocular.util.ArrayHelper;

public class ArrayHelperTests {

	@Test
	public void test_prepend() {
		{
			int[] b = ArrayHelper.prepend(0, new int[] { 1, 2, 3 });
			assertEquals(4, b.length);
			assertEquals(0, b[0]);
			assertEquals(1, b[1]);
			assertEquals(2, b[2]);
			assertEquals(3, b[3]);
		}
		{
			int[] b = ArrayHelper.prepend(0, new int[] {});
			assertEquals(1, b.length);
			assertEquals(0, b[0]);
		}
	}

	@Test
	public void test_append() {
		{
			Integer[] b = ArrayHelper.append(new Integer[] { 0, 1, 2 }, 3);
			assertEquals(4, b.length);
			assertEquals((int) 0, (int) b[0]);
			assertEquals((int) 1, (int) b[1]);
			assertEquals((int) 2, (int) b[2]);
			assertEquals((int) 3, (int) b[3]);
		}
		{
			Integer[] b = ArrayHelper.append(new Integer[] {}, 0);
			assertEquals(1, b.length);
			assertEquals((int) 0, (int) b[0]);
		}
	}

	@Test
	public void test_take() {
		{
			int[] b = ArrayHelper.take(new int[] { 1, 2, 3 }, 2);
			assertEquals(2, b.length);
			assertEquals(1, b[0]);
			assertEquals(2, b[1]);
		}
		{
			int[] b = ArrayHelper.take(new int[] { 1, 2, 3 }, 3);
			assertEquals(3, b.length);
			assertEquals(1, b[0]);
			assertEquals(2, b[1]);
			assertEquals(3, b[2]);
		}
		{
			int[] b = ArrayHelper.take(new int[] { 1, 2, 3 }, 0);
			assertEquals(0, b.length);
		}
		{
			int[] b = ArrayHelper.take(new int[] { 1, 2, 3 }, 8);
			assertEquals(3, b.length);
			assertEquals(1, b[0]);
			assertEquals(2, b[1]);
			assertEquals(3, b[2]);
		}
		{
			int[] b = ArrayHelper.take(new int[] {}, 0);
			assertEquals(0, b.length);
		}
		{
			int[] b = ArrayHelper.take(new int[] {}, 2);
			assertEquals(0, b.length);
		}
	}

	@Test
	public void test_takeRight() {
		{
			int[] b = ArrayHelper.takeRight(new int[] { 1, 2, 3 }, 2);
			assertEquals(2, b.length);
			assertEquals(2, b[0]);
			assertEquals(3, b[1]);
		}
		{
			int[] b = ArrayHelper.takeRight(new int[] { 1, 2, 3 }, 3);
			assertEquals(3, b.length);
			assertEquals(1, b[0]);
			assertEquals(2, b[1]);
			assertEquals(3, b[2]);
		}
		{
			int[] b = ArrayHelper.takeRight(new int[] { 1, 2, 3 }, 0);
			assertEquals(0, b.length);
		}
		{
			int[] b = ArrayHelper.takeRight(new int[] { 1, 2, 3 }, 8);
			assertEquals(3, b.length);
			assertEquals(1, b[0]);
			assertEquals(2, b[1]);
			assertEquals(3, b[2]);
		}
		{
			int[] b = ArrayHelper.takeRight(new int[] {}, 0);
			assertEquals(0, b.length);
		}
		{
			int[] b = ArrayHelper.takeRight(new int[] {}, 2);
			assertEquals(0, b.length);
		}
	}
}
