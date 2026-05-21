import pathlib
from jinja2 import Environment, FileSystemLoader
from playwright.async_api import async_playwright


async def generate_report_pdf(data, template_name="Report.html", output_name="analysis_report.pdf"):
    script_dir = pathlib.Path(__file__).parent.resolve()

    current_dir = (script_dir / ".." / "ReportTemplates").resolve()
    temp_html_path = current_dir / "temp_render.html"

    print(f"Searching for templates in: {current_dir}")

    env = Environment(loader=FileSystemLoader(str(current_dir)))

    try:
        template = env.get_template(template_name)
        rendered_content = template.render(data)

        temp_html_path.write_text(rendered_content, encoding="utf-8")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(bypass_csp=True)
            page = await context.new_page()

            await page.goto(temp_html_path.as_uri(), wait_until="networkidle")
            await page.pdf(path=output_name, format="A4", print_background=True)
            await browser.close()

        print(f"Successfully generated: {output_name}")
        return output_name
    except Exception as e:
        print(f"Error generating PDF: {e}")
        raise e
    finally:
        if temp_html_path.exists():
            temp_html_path.unlink()