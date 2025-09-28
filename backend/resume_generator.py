def generate_ats_resume_text(resume_data):
    """
    Generates a plain-text, single-column resume string optimized for ATS parsers.

    Args:
        resume_data (dict): A dictionary containing structured resume information.
                            Example structure:
                            {
                                "name": "John Doe",
                                "email": "john.doe@email.com",
                                "phone": "555-123-4567",
                                "linkedin": "linkedin.com/in/johndoe",
                                "summary": "Experienced software engineer...",
                                "skills": ["Python", "JavaScript", "SQL"],
                                "experience": [
                                    {
                                        "title": "Software Engineer",
                                        "company": "Tech Corp",
                                        "dates": "2020-2023",
                                        "bullets": ["Developed web applications"]
                                    }
                                ],
                                "education": [
                                    {
                                        "degree": "BS Computer Science",
                                        "school": "University Name",
                                        "dates": "2016-2020"
                                    }
                                ]
                            }
    Returns:
        str: A formatted string ready to be saved as a.txt or.docx file.
    """
    output = []

    # --- Contact Information ---
    output.append(resume_data.get("name", ""))
    contact_info = " | ".join(filter(None, [
        resume_data.get("email"),
        resume_data.get("phone"),
        resume_data.get("linkedin")
    ]))
    output.append(contact_info)
    output.append("\n" + "="*80 + "\n")

    # --- Summary ---
    if resume_data.get("summary"):
        output.append("SUMMARY")
        output.append(resume_data["summary"])
        output.append("\n")

    # --- Skills ---
    if resume_data.get("skills"):
        output.append("SKILLS")
        output.append(", ".join(resume_data["skills"]))
        output.append("\n")

    # --- Work Experience ---
    if resume_data.get("experience"):
        output.append("WORK EXPERIENCE")
        for job in resume_data["experience"]:
            output.append(f"\n{job.get('title', '')}")
            output.append(f"{job.get('company', '')} | {job.get('dates', '')}")
            for bullet in job.get("bullets", []):
                output.append(f"- {bullet}")
        output.append("\n")

    # --- Education ---
    if resume_data.get("education"):
        output.append("EDUCATION")
        for edu in resume_data["education"]:
            output.append(f"\n{edu.get('degree', '')}")
            output.append(f"{edu.get('school', '')} | {edu.get('dates', '')}")

    return "\n".join(output)