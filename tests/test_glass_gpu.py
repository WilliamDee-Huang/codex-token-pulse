"""Real shader compilation and blur behavior when OpenGL 3.3 is available."""
from pathlib import Path
import unittest

try:
    import moderngl
except ImportError:
    moderngl = None

from monitor_refraction import VERTEX


@unittest.skipIf(moderngl is None, 'ModernGL is optional')
class GlassGpuTests(unittest.TestCase):
    def test_blur_diffuses_background_detail_and_material_compiles(self):
        try:
            context = moderngl.create_context(standalone=True, require=330)
        except Exception as exc:
            self.skipTest(f'OpenGL 3.3 unavailable: {exc}')
        resources = []
        try:
            assets = Path(__file__).resolve().parents[1] / 'assets'
            material = context.program(vertex_shader=VERTEX,
                                       fragment_shader=(assets / 'liquid-glass.frag').read_text())
            resources.append(material)
            blur = context.program(vertex_shader=VERTEX,
                                   fragment_shader=(assets / 'glass-blur.frag').read_text())
            resources.append(blur)
            width, height = 64, 4
            source = context.texture((width, height), 4, b''.join(
                bytes((255, 255, 255, 255)) if x == 32 else bytes((0, 0, 0, 255))
                for _y in range(height) for x in range(width)))
            resources.append(source)
            source.repeat_x = source.repeat_y = False
            source.filter = (moderngl.LINEAR, moderngl.LINEAR)
            target = context.simple_framebuffer((width, height), components=4)
            resources.append(target)
            vao = context.vertex_array(blur, [])
            resources.append(vao)
            source.use(0)
            blur['sourceImage'] = 0
            blur['targetSize'] = (width, height)
            blur['stepUV'] = (1 / width, 0)
            blur['sigma'] = 3
            target.use()
            context.viewport = (0, 0, width, height)
            vao.render(vertices=3)
            result = target.read(components=4)
            self.assertLess(result[32 * 4], 60, 'Blur should suppress the sharp stripe')
            self.assertGreater(result[29 * 4], 10, 'Blur should spread into neighboring pixels')
            self.assertEqual(result[0], 0, 'Distant background should remain unchanged')
        finally:
            for resource in reversed(resources):
                resource.release()
            context.release()


if __name__ == '__main__':
    unittest.main()
