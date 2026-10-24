from google.cloud import documentai_v1 as docai
from google.api_core.client_options import ClientOptions
import os

os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS", "google-cred/docai-key.json")
processor_name = os.environ.get(
    "GOOGLE_DOCAI_PROCESSOR",
    "projects/your-project-id/locations/us/processors/your-processor-id",
)
sample_pdf = os.environ.get("SAMPLE_PDF_PATH", "uploads/sample.pdf")

if __name__ == "__main__":
    opts = ClientOptions(api_endpoint="us-documentai.googleapis.com")
    client = docai.DocumentProcessorServiceClient(client_options=opts)
    request = docai.ProcessRequest(
        name=processor_name,
        raw_document=docai.RawDocument(
            content=open(sample_pdf, "rb").read(4000000),
            mime_type="application/pdf",
        ),
    )
    result = client.process_document(request=request)
    for p in result.document.pages:
        print(f"Page: {p.page_number}")
        for ve in p.visual_elements:
            print(f"  VisualElement: type={ve.type_}")
        for b in p.blocks:
            print(
                f"  Block text: {result.document.text[b.layout.text_anchor.text_segments[0].start_index:b.layout.text_anchor.text_segments[0].end_index][:20]}"
            )
