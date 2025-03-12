// The project function defines how your document looks.
// It takes your content and some metadata and formats it.
// Go ahead and customize it to your liking!
#let front_page(title: "", authors: (), numbers: (), date: none) = {
  // Set the document's basic properties.
  set document(author: authors, title: title)

  // Title row.
  align(horizon + center)[
    #block(text(weight: 700, 1.9em, title))
    #v(1em, weak: true)
    #date
    #grid(
      columns: authors.len(),
      gutter: 1em,
      ..authors.map(author => align(center, strong(author))),
      ..numbers.map(number => align(center, number)),
    )
  ]
}
