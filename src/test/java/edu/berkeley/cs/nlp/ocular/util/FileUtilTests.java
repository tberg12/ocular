package edu.berkeley.cs.nlp.ocular.util;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;

import org.junit.Test;

/**
 * @author Dan Garrette (dhgarrette@gmail.com)
 */
public class FileUtilTests {

	@Test
	public void test_lowestCommonPath() {
		{
		List<String> paths = new ArrayList<String>();
		paths.add("/well/this/and/that/");
		paths.add("/well/this/and/the/other.txt");
		paths.add("/well/this/and/thus.txt");
		String lcpd = FileUtil.lowestCommonPath(paths);
		assertEquals("/well/this/and", lcpd);
		}
		{
		List<String> paths = new ArrayList<String>();
		paths.add("/well/this/and/thus.txt");
		String lcpd = FileUtil.lowestCommonPath(paths);
		assertEquals("/well/this/and/thus.txt", lcpd);
		}
		{
		List<String> paths = new ArrayList<String>();
		paths.add("/well/this/and/");
		paths.add("/well/this/and/");
		String lcpd = FileUtil.lowestCommonPath(paths);
		assertEquals("/well/this/and", lcpd);
		}
		{
		List<String> paths = new ArrayList<String>();
		paths.add("/well/this/and/");
		String lcpd = FileUtil.lowestCommonPath(paths);
		assertEquals("/well/this/and", lcpd);
		}
	}

	@Test
	public void test_pathRelativeTo() {
		String d0 = "/well/this/and/";
		String f1 = "/well/this/and/that.txt";
		String f2 = "/well/this/and/that";
		String f3 = "/well/this/and/that/or.txt";
		String f4 = "/well/this/and/that/else/";
		
		assertEquals("that.txt", FileUtil.pathRelativeTo(f1,d0));
		assertEquals("that", FileUtil.pathRelativeTo(f2,d0));
		assertEquals("that/or.txt", FileUtil.pathRelativeTo(f3,d0));
		assertEquals("that/else", FileUtil.pathRelativeTo(f4,d0));
	}

}
